import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from random import randint
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from gan_protein_structural_requirements.models import networks
from gan_protein_structural_requirements.utils.protein_visualizer import esm_predict_api_batch, jupy_viz_obj
from gan_protein_structural_requirements.models import random_sample_protein, process_outs
import gan_protein_structural_requirements.models.metrics as m


class CCC_ProGAN_Model(nn.Module):
    def __init__(self, config):
        super(CCC_ProGAN_Model, self).__init__()

        self.config = config
        self.hyperparams = self.config.hyperparams
        self.latent_dim = self.hyperparams.latent_dim
        self.generator_hidden_dims = self.hyperparams.net_G_hidden_dims
        self.discriminator_hidden_dims = self.hyperparams.net_D_hidden_dims
        self.objective_dim = self.config.objective_dim
        self.seq_length = self.config.seq_length
        self.vocab_size = self.config.vocab_size
        self.net_r_save_path = self.config.r_path

        self.seq_loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab_size - 1) #ignore padding (mask loss)
        self.obj_loss_fn = nn.MSELoss()
        self.adversarial_loss_fn = nn.BCELoss()

        if config.mode == "CNN":

            self.net_G = networks.CNNGenerator(self.latent_dim, 
                                            self.objective_dim, 
                                            self.seq_length, 
                                            self.vocab_size, 
                                            self.generator_hidden_dims)

            self.net_G.train()
            
            self.net_G.apply(self.net_G._init_weights)
            
            self.net_D = networks.Discriminator(self.seq_length,
                                                self.discriminator_hidden_dims,
                                                self.objective_dim,
                                                self.vocab_size)
            
            self.net_D.train()
        
        elif config.mode == "Transformer":

            self.net_G = networks.TransformerGenerator(self.latent_dim,
                                                       self.objective_dim,
                                                       self.seq_length,
                                                       self.vocab_size,
                                                       self.generator_hidden_dims)
            
            self.net_G.train()
            
            self.net_D = networks.ConvolutionalDiscriminator(self.discriminator_hidden_dims,
                                                             self.seq_length,
                                                             self.objective_dim,
                                                             self.vocab_size)
            
            self.net_D.train()


        self.net_R = networks.SeqToVecEnsemble(self.vocab_size, 
                                               self.seq_length)
        self.net_R.load_state_dict(torch.load(self.net_r_save_path))
        self.net_R.eval()


    def set_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def print_networks(self):

        print(self.net_G)
        print(self.net_D)
        print(self.net_R)

        total_params_G = sum(
	        param.numel() for param in self.net_G.parameters()
        )

        total_params_D = sum(
	        param.numel() for param in self.net_D.parameters()
        )

        total_params_R = sum(
	        param.numel() for param in self.net_R.parameters()
        )

        print("Trainable Parameters: ", str(total_params_G + total_params_D))
        print("Non-trainable Parameters: ", str(total_params_R))
        print("Total Parameters: ", str(total_params_G + total_params_D + total_params_R))


    def save_networks(self, save_dir, epoch, save_g=True, save_d=False):

        #create file suffixes
        suffix_g = f"g_epoch_{epoch}"
        suffix_d = f"d_epoch_{epoch}"

        #get save directories
        save_dir_g = os.path.join(save_dir, suffix_g)
        save_dir_d = os.path.join(save_dir, suffix_d)

        if save_g:
            torch.save(self.net_G.state_dict(), save_dir_g)
        
        if save_d:
            torch.save(self.net_D.state_dict(), save_dir_d)


    def train(self, train_config):
        
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),train_config.net_G_lr)
        self.optimizer_D = torch.optim.SGD(self.net_D.parameters(),train_config.net_D_lr, momentum=train_config.momentum_D)

        scheduler_G = ExponentialLR(self.optimizer_G, train_config.decay_G)
        scheduler_D = ExponentialLR(self.optimizer_D, train_config.decay_D)


        train_data = DataLoader(dataset=train_config.train_dataset,
                                batch_size=train_config.batch_size)
        
        val_data = DataLoader(dataset=train_config.val_dataset,
                              batch_size=len(train_config.val_dataset))
        
        #print parameters and begin training
        self.print_networks()

        #set up train directories
        save_dir = os.path.join(train_config.root_dir, "checkpoints", train_config.model_name)
        log_dir = os.path.join(train_config.root_dir, "checkpoints", train_config.model_name, "logs")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        #setup tensorboard logging
        writer = SummaryWriter(log_dir=log_dir)

        print()
        print()
        print("|----------Beginning Training---------|")
        print()
        print()

        for epoch in range(1, train_config.epochs + 1):

            running_loss_d = 0
            running_loss_g = 0
            running_loss_obj = 0
            running_loss_seq = 0
            running_loss_gan = 0
            d_iters = 0
            g_iters = 0

            print(f"Training Discriminator for {train_config.D_iters} Iters")
            for k in range(train_config.D_iters):
                for iter, data in enumerate(train_data, 0):

                    self.net_D.train()
                    #set important input data
                    X = data["X"]
                    Y = data["Y"]
                    ids = data["IDS"]
                    latent = torch.randn(X.size(0),self.latent_dim)
                    real_label = torch.ones((X.size(0),1))
                    fake_label = torch.zeros((X.size(0),1))

                    #train D
                    self.net_D.zero_grad()

                    d_pred_real = self.net_D(Y.long().permute(0,2,1), X.permute(0,2,1))
                    fake = F.gumbel_softmax(self.net_G(latent, X[:,0,:]))
                    d_pred_fake = self.net_D(fake.long().detach(),X.permute(0,2,1))

                    #discriminator loss
                    loss_real = self.adversarial_loss_fn(d_pred_real, real_label)
                    loss_fake = self.adversarial_loss_fn(d_pred_fake, fake_label)
                    loss_d = ((loss_real + loss_fake) / 2) * train_config.lambda_D
                    loss_d.backward()

                    self.optimizer_D.step()

                    running_loss_d += loss_d.item()

                    d_iters += 1

                    if iter % train_config.print_freq == 0:
                        print(f"[{epoch}/{train_config.epochs}][{k+1}/{train_config.D_iters}][{iter}/{len(train_data)}]\tLoss_D: {loss_d}")


            print()
            print()
            print(f"Training Generator for {train_config.G_iters} Iters")
            for k in range(train_config.G_iters):
                for iter, data in enumerate(train_data, 0):
                    self.net_G.train()
                    #set important input data
                    X = data["X"]
                    Y = data["Y"]
                    ids = data["IDS"]
                    latent = torch.randn(X.size(0),self.latent_dim)
                    real_label = torch.ones((X.size(0),1))
                    fake_label = torch.zeros((X.size(0),1))

                    self.set_requires_grad(self.net_R)
                    self.net_G.zero_grad()

                    fake = self.net_G(latent, X[:,0,:])

                    #gan loss
                    d_pred_fake = self.net_D(fake.long(),X.permute(0,2,1))
                    gan_loss = self.adversarial_loss_fn(d_pred_fake, real_label) * train_config.lambda_G

                    #sequence loss
                    seq_loss = self.seq_loss_fn(fake,torch.argmax(Y.float(),dim=-1)) * train_config.lambda_seq

                    #reconstruction loss
                    fake_onehot = F.gumbel_softmax(fake,tau=1,hard=True)
                    sec, pol = self.net_R(fake_onehot)
                    fake_objectives = torch.cat([sec,pol],dim=-1)
                    obj_loss = self.obj_loss_fn(fake_objectives,X[:,0,:].float()) * train_config.lambda_obj

                    loss_g = gan_loss + obj_loss + seq_loss
                    loss_g.backward()

                    self.optimizer_G.step()

                    g_iters += 1

                    running_loss_g += loss_g
                    running_loss_seq += seq_loss
                    running_loss_obj += obj_loss
                    running_loss_gan += gan_loss

                    if iter % train_config.print_freq == 0:
                        print(f"[{epoch}/{train_config.epochs}][{k+1}/{train_config.G_iters}][{iter}/{len(train_data)}]\tLoss_G: {loss_g}\tLoss_Seq: {seq_loss}\tLoss_Obj: {obj_loss}\tLoss_GAN: {gan_loss}")
            
            scheduler_D.step()
            scheduler_G.step()

            print()
            print()
            #add avg losses to tb
            avg_loss_G = running_loss_g/g_iters
            avg_loss_D = running_loss_d/d_iters
            avg_loss_seq = running_loss_seq/g_iters
            avg_loss_obj = running_loss_obj/g_iters
            avg_loss_gan = running_loss_gan/g_iters
            writer.add_scalar("train/loss/G",avg_loss_G, global_step=epoch)
            writer.add_scalar("train/loss/D",avg_loss_D, global_step=epoch)
            writer.add_scalar("train/loss/gan", avg_loss_gan, global_step=epoch)
            writer.add_scalar("train/loss/seq",avg_loss_seq, global_step=epoch)
            writer.add_scalar("train/loss/obj",avg_loss_obj, global_step=epoch)
            

            print(f"|-------------Validation Step At End of Epoch {epoch}--------------|")

            #at the end of each epoch perform validation
            with torch.no_grad():

                g_loss_val = []
                d_loss_val = []
                seq_loss_val = []
                obj_loss_val = []
                gan_loss_val = []

                for _, data in enumerate(val_data, 0):

                    #unpack and set inputs
                    X = data["X"]
                    Y = data["Y"]
                    ids = data["IDS"]
                    latent = torch.randn(X.size(0),self.latent_dim)
                    real_label = torch.ones((X.size(0),1))
                    fake_label = torch.zeros((X.size(0),1))

                    #compute outputs on validation data
                    g_pred = self.net_G(latent, X[:,0,:])

                    g_pred_onehot = F.gumbel_softmax(g_pred, tau=1, hard=True)
                    sec, pol = self.net_R(g_pred_onehot)
                    x_pred = torch.cat([sec,pol],dim=-1)

                    d_pred_real = self.net_D(Y.long().permute(0,2,1), X.permute(0,2,1))
                    d_pred_fake = self.net_D(g_pred.long(),X.permute(0,2,1))

                    #calculate losses
                    obj_loss = self.obj_loss_fn(x_pred,X[:,0,:].float())
                    seq_loss = self.seq_loss_fn(g_pred,torch.argmax(Y.float(), dim=-1))
                    adv_loss = self.adversarial_loss_fn(d_pred_fake, real_label)
                    g_loss = obj_loss + adv_loss + seq_loss
                    d_loss_real = self.adversarial_loss_fn(d_pred_real, real_label)
                    d_loss_fake = self.adversarial_loss_fn(d_pred_fake, fake_label)
                    d_loss = (d_loss_real + d_loss_fake)/2

                    gan_loss_val.append(adv_loss.item())
                    g_loss_val.append(g_loss.item())
                    d_loss_val.append(d_loss.item())
                    seq_loss_val.append(seq_loss.item())
                    obj_loss_val.append(obj_loss.item())

                    #sequences = process_outs(g_pred.permute(0,2,1), train_config.train_dataset.decode_cats)


            writer.add_scalar("val/loss/G", g_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/D", d_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/gan", gan_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/seq", seq_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/obj", obj_loss_val[0], global_step=epoch)

            with torch.no_grad():
                #create random proteins
                objectives, latent = random_sample_protein(train_config.num_val_prot, self.latent_dim)

                #pass through generator
                out = self.net_G(latent, objectives)
                out = out.permute(0,2,1)

                #compute sequences
                if epoch % train_config.viz_freq == 0:
                    sequences = process_outs(out, train_config.train_dataset.decode_cats)

                    #get pdb from esmfold
                    pdbs = esm_predict_api_batch(sequences)

                #create protein visualization for first generated protein in set of sequences
                    jupy_viz_obj(pdbs[0])
            
            print(f"Loss_G: {g_loss_val[0]}\tLoss_D: {d_loss_val[0]}\tLoss_seq: {seq_loss_val[0]}\tLoss_obj: {obj_loss_val[0]}")
            print()
            print()
            
            #save models
            if epoch % train_config.save_freq == 0:
                self.save_networks(save_dir, epoch)