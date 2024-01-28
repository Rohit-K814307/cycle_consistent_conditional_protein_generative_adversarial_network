from gan_protein_structural_requirements.models import networks
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from random import randint
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from gan_protein_structural_requirements import test
from gan_protein_structural_requirements.utils.protein_visualizer import esm_predict_api_batch, viz_protein_seq, jupy_viz_obj
from gan_protein_structural_requirements.models import random_sample_protein, process_outs
from gan_protein_structural_requirements.models.metrics import pairwise_avg_rmsd
from gan_protein_structural_requirements.models import metrics as m



class RNN_CCCProGANModel(nn.Module):
    def __init__(self, latent_dim, objective_dim, hidden_dim, num_rnn_layers, sequence_dim, batch_size, max_prot_len, lr, lr_beta, epsilon, lambda_seq, lambda_obj, lambda_g, lambda_d, weight_decay, cos_eps, path_to_r, path_to_esm, map, viz=True):
        """Initialize a RNN-based CCC ProGAN model
        
        Parameters:

            latent_dim -- last dimension of latent space 

            objective_dim -- number of objectives

            hidden_dim -- hidden dimensions for generator

            num_rnn_layers -- number of rnn layers for generator

            sequence_dim -- vocab size of sequences

            batch_size -- batch size of dataset

            max_prot_len -- maximum length of proteins in dataset

            lr -- learning rate of models

            lr_beta -- learning rate beta of models

            epsilon -- epsilon of models

            lambda_seq -- bias for sequence loss

            lambda_obj -- bias for objective loss

            lambda_g -- bias for generator (individual) loss

            lambda_d -- bias for discriminator loss

            path_to_r -- path to seqtovec (pretrained) model

            path_to_esm -- path to esmfold model

            viz -- load in esm models etc.
        
        """

        super(RNN_CCCProGANModel, self).__init__()

        self.batch_size = batch_size
        self.max_prot_len = max_prot_len
        self.latent_dim = latent_dim
        self.lambda_seq = lambda_seq
        self.lambda_obj = lambda_obj
        self.lambda_g = lambda_g
        self.lambda_d = lambda_d
        self.map = map

        self.net_g = networks.RNN_generator(latent_dim,objective_dim,hidden_dim,num_rnn_layers,sequence_dim).train()
        self.net_d = networks.Discriminator(batch_size, max_prot_len, [64, 32],objective_dim,sequence_dim).train()

        self.net_r = networks.SeqToVecEnsemble(sequence_dim, max_prot_len)
        self.net_r.load_state_dict(torch.load(path_to_r))
        self.net_r.eval()
        self.set_requires_grad([self.net_r],False)

        self.score_names = ["seq_feasibility", "seq_diversity"]
        self.loss_names = ["loss_seq", "loss_obj", "loss_gan", "loss_g", "loss_d"]
        self.networks = ["net_g","net_d","net_r"]

        self.seq_loss_fn = nn.CrossEntropyLoss(ignore_index=21)
        self.obj_loss_fn = nn.MSELoss()
        self.gan_loss_fn = nn.BCELoss()

        self.optimizer_g = torch.optim.RMSprop(self.net_g.parameters(),lr=lr[0], eps=epsilon[0], weight_decay=weight_decay[0])
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),lr=lr[1], betas=(lr_beta[0], 0.999), eps=epsilon[0], weight_decay=weight_decay[1])
        self.optimizers = [self.optimizer_g, self.optimizer_d]
    

    def set_input(self, dataset):
        """Unpack input data and perform data allocation steps

        Parameters:

            dataloader (dict): include data itself and metadata information

        """

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)


    def set_iter_input(self, dataloader):

        self.X = dataloader["X"]

        self.Y = dataloader["Y"]

        self.ids = dataloader["IDS"]


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        
        Parameters:
            
            nets (network list)   - a list of networks
            
            requires_grad (bool)  - whether the networks require gradients or not
        
        """

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def forward_g(self):

        #create latent input 
        latent_input = torch.randn(self.X.size(0), self.max_prot_len, self.latent_dim)

        #create generated sequences
        self.gen_seq = self.net_g(latent_input, self.X)

        #find predicted objectives
        self.r_x = F.gumbel_softmax(self.gen_seq,tau=1,hard=True)
        self.sec, self.pol = self.net_r(self.r_x.float().permute(0,2,1))
        self.x_hat = torch.cat([self.sec, self.pol], dim=-1)
        

        #calculate score metrics
        with torch.no_grad():

            preds = self.r_x.detach().clone()

            self.seq_feasibility = m.seq_feasibility(preds, self.Y.detach().clone())

            self.seq_diversity = test.get_metrics_rand(self.net_g,20,self.map,self.max_prot_len,self.latent_dim)["Diversity"]


    def forward_d(self):
        self.real_labels = torch.ones(self.X.size(0), 1)
        self.fake_labels = torch.zeros(self.X.size(0), 1)
        self.disc_output_real = self.net_d(self.Y.long(), self.X)
        self.disc_output_fake = self.net_d(self.net_g(torch.randn(self.X.size(0),self.max_prot_len,self.latent_dim),self.X).detach().long(), self.X)


    def backward_d(self):
        loss_d_real = self.gan_loss_fn(self.disc_output_real, self.real_labels) * self.lambda_d
        loss_d_fake = self.gan_loss_fn(self.disc_output_fake, self.fake_labels) * self.lambda_d

        self.loss_d = (loss_d_real + loss_d_fake) / 2 #optimizable
        self.loss_d.backward()


    def backward_g(self):
        #calculate gan losses
        self.loss_obj = self.obj_loss_fn(self.x_hat, self.X[:,0,:].float()) * self.lambda_obj
        self.loss_seq = self.seq_loss_fn(self.gen_seq.permute(0,2,1), torch.argmax(self.Y.float(),dim=-1)) * self.lambda_seq #square of cos similarity to get rid of potential negatives since range of cos is [-1,1]
        self.loss_gan = self.gan_loss_fn(self.net_d(self.gen_seq.long(), self.X), torch.ones(self.X.size(0), 1)) * self.lambda_g

        #summed entire gan loss
        self.loss_g =  self.loss_obj + self.loss_seq + self.loss_gan    #optimizable

        self.loss_g.backward()

    
    def optimize_parameters(self, epoch, epochs_g, epochs_d):

        print()
        print()
        print("|---------------------Train Discriminator----------------|")

        for iter_epoch in range(epochs_d):
            for iter, minibatch in enumerate(self.dataloader):

                self.set_iter_input(minibatch)

                #optimize discriminator
                self.optimizer_d.zero_grad()
                self.forward_d()
                self.backward_d()
                self.optimizer_d.step()

            print(f"|-------------Metrics for Discriminator: Epoch {epoch}; iter {iter_epoch}-----------|")
            print("Discriminator loss: " + str(self.loss_d.item()))
            print()
            print()


        print("|----------------------Train Generator-------------------|")


        for iter_epoch in range(epochs_g):
            for iter, minibatch in enumerate(self.dataloader):

                self.set_iter_input(minibatch)

                #optimize generator
                self.set_requires_grad([self.net_r],False)
                self.optimizer_g.zero_grad()
                self.forward_g()
                self.backward_g()
                self.optimizer_g.step()
            
            print(f"|-------------Metrics for Generator: Epoch {epoch}; iter {iter_epoch}-----------|")
            print("Generator Summed Loss: " + str(self.loss_g.item()))
            print("Sequence CCE Loss: " + str(self.loss_seq.item()))
            print("Objective Loss: " + str(self.loss_obj.item()))
            print("Gan Loss: " + str(self.loss_gan.item()))
            print(self.num_to_seq())
            print()
            print()



    def save_model(self, save_dir, epoch):
        """
        
        Parameters
        
            epoch -- epoch of current training step

            iter -- iteration of current training step

            save_dir -- directory to save model to
            

        """

        #create file suffixes
        suffix_g = f"g_epoch_{epoch}"
        suffix_d = f"d_epoch_{epoch}"

        #get save directories
        save_dir_g = os.path.join(save_dir, suffix_g)
        save_dir_d = os.path.join(save_dir, suffix_d)

        #save models
        torch.save(self.net_g.state_dict(), save_dir_g)
        torch.save(self.net_d.state_dict(), save_dir_d)




    def num_to_seq(self):
        with torch.no_grad(): #run in no grad to make sure it doesn't affect gradients

            #detach onehot vector and extract actual sequences
            output = self.r_x.detach().clone()

            sequences = []

            for sequence in output:
                seq = ""
                for residue in sequence:
                    seq += self.map[torch.argmax(residue,dim=-1).item()]

                seq = seq.replace("-", "")
                sequences.append(seq)

        return sequences


    def compute_pdb(self):

        with torch.no_grad(): #run in no grad to make sure it doesn't affect gradients

            #detach onehot vector and extract actual sequences
            output = self.r_x.detach().clone()

            sequences = []

            for sequence in output:
                seq = ""
                for residue in sequence:
                    seq += self.map[torch.argmax(residue,dim=-1).item()]

                seq = seq.replace("-", "")
                sequences.append(seq)

            pdbs, ids = esm_predict_api_batch(sequences[0:3]), self.ids[0:3]

        return pdbs, ids


    def get_viz(self):
        pdbs, ids = self.compute_pdb()

        images = []
        for pdb in pdbs:
            images.append(viz_protein_seq(pdb))
            time.sleep(0.5)

        return images, ids



###########################################################################################

class CNN_CCC_ProGAN_Model(nn.Module):
    def __init__(self, config):
        super(CNN_CCC_ProGAN_Model, self).__init__()

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

        self.net_G = networks.CNNGenerator(self.latent_dim, 
                                           self.objective_dim, 
                                           self.seq_length, 
                                           self.vocab_size, 
                                           self.generator_hidden_dims)
        
        self.net_G.apply(self.net_G._init_weights)
        
        self.net_D = networks.Discriminator(self.seq_length,
                                            self.discriminator_hidden_dims,
                                            self.objective_dim,
                                            self.vocab_size)

        self.net_R = networks.SeqToVecEnsemble(self.vocab_size, 
                                               self.seq_length)
        self.net_R.load_state_dict(torch.load(self.net_r_save_path))
        self.net_R.eval()


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

        self.optimizer_G = torch.optim.SGD(self.net_G.parameters(),train_config.net_G_lr, momentum=train_config.momentum_G)
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
            d_iters = 0
            g_iters = 0

            print(f"Training Discriminator for {train_config.D_iters} Iters")
            for k in range(train_config.D_iters):
                for iter, data in enumerate(train_data, 0):
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
                    #set important input data
                    X = data["X"]
                    Y = data["Y"]
                    ids = data["IDS"]
                    latent = torch.randn(X.size(0),self.latent_dim)
                    real_label = torch.ones((X.size(0),1))
                    fake_label = torch.zeros((X.size(0),1))

                    self.net_R.zero_grad()
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

                    if iter % train_config.print_freq == 0:
                        print(f"[{epoch}/{train_config.epochs}][{k+1}/{train_config.G_iters}][{iter}/{len(train_data)}]\tLoss_G: {loss_g}\tLoss_Seq: {seq_loss}\tLoss_Obj: {obj_loss}")
            
            scheduler_D.step()
            scheduler_G.step()

            print()
            print()
            #add avg losses to tb
            avg_loss_G = running_loss_g/g_iters
            avg_loss_D = running_loss_d/d_iters
            avg_loss_seq = running_loss_seq/g_iters
            avg_loss_obj = running_loss_obj/g_iters
            writer.add_scalar("train/loss/G",avg_loss_G, global_step=epoch)
            writer.add_scalar("train/loss/D",avg_loss_D, global_step=epoch)
            writer.add_scalar("train/loss/seq",avg_loss_seq, global_step=epoch)
            writer.add_scalar("train/loss/obj",avg_loss_obj, global_step=epoch)
            

            print(f"|-------------Validation Step At End of Epoch {epoch}--------------|")

            #at the end of each epoch perform validation
            with torch.no_grad():

                g_loss_val = []
                d_loss_val = []
                seq_loss_val = []
                obj_loss_val = []

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

                    g_loss_val.append(g_loss.item())
                    d_loss_val.append(d_loss.item())
                    seq_loss_val.append(seq_loss.item())
                    obj_loss_val.append(obj_loss.item())

                    sequences = process_outs(g_pred.permute(0,2,1), train_config.train_dataset.decode_cats)

            writer.add_scalar("val/loss/G", g_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/D", d_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/seq", seq_loss_val[0], global_step=epoch)
            writer.add_scalar("val/loss/obj", obj_loss_val[0], global_step=epoch)

            with torch.no_grad():
                #create random proteins
                objectives, latent = random_sample_protein(train_config.num_val_prot, self.latent_dim)

                #pass through generator
                out = self.net_G(latent, objectives)
                out = out.permute(0,2,1)

                #compute sequences
                sequences = process_outs(out, train_config.train_dataset.decode_cats)

                #get pdb from esmfold
                pdbs = esm_predict_api_batch(sequences)

                # #get metrics
                # rmsd = pairwise_avg_rmsd(pdbs)

                # #add scores to tb
                # writer.add_scalar("val/metric/rmsd", rmsd, global_step=epoch)

                #create protein visualization for first generated protein in set of sequences
                if epoch % train_config.viz_freq == 0:
                    jupy_viz_obj(pdbs[0])
            
            print(f"Loss_G: {g_loss_val[0]}\tLoss_D: {d_loss_val[0]}\tLoss_seq: {seq_loss_val[0]}\tLoss_obj: {obj_loss_val[0]}")#\tRMSD: {rmsd}")
            print()
            print()
            
            #save models
            if epoch % train_config.save_freq == 0:
                self.save_networks(save_dir, epoch)