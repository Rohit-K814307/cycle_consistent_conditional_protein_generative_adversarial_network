from gan_protein_structural_requirements.models import networks
import sys, os
import torch
import itertools
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn

class RNN_CCCProGANModel(nn.Module):
    def __init__(self, latent_dim, objective_dim, hidden_dim, num_rnn_layers, sequence_dim, batch_size, max_prot_len, lr, lr_beta, epsilon, path_to_r):
        """Initialize a RNN-based CCC_ProGAN model
        
        Parameters
        
        """


        super(RNN_CCCProGANModel, self).__init__()

        self.batch_size = batch_size
        self.max_prot_len = max_prot_len
        self.latent_dim = latent_dim
        self.real_labels = torch.ones(self.batch_size, 1)
        self.fake_labels = torch.zeros(self.batch_size, 1)

        self.net_g = networks.RNN_generator(latent_dim,objective_dim,hidden_dim,num_rnn_layers,sequence_dim)
        self.net_d = networks.Discriminator(batch_size, max_prot_len, discrminator_layer_sizes,objective_dim,sequence_dim)
        self.net_r = networks.SeqToVecEnsemble(sequence_dim, max_prot_len).load_state_dict(torch.load(path_to_r))

        self.score_names = []
        self.loss_names = ["loss_seq", "loss_obj", "loss_gan", "loss_g", "loss_d"]
        self.visual_names = ["gen_seq"]
        self.networks = ["net_g","net_d","net_r"]

        self.seq_loss_fn = nn.CrossEntropyLoss()
        self.obj_loss_fn = nn.MSELoss()
        self.gan_loss_fn = nn.BCELoss()
        self.d_loss_fn = nn.BCELoss()

        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),lr=lr, betas=(lr_beta, 0.999), eps=epsilon, weight_decay=0)
        self.optimizer_d =torch.optim.Adam(self.net_d.parameters(),lr=lr, betas=(lr_beta, 0.999), eps=epsilon, weight_decay=0)


    def set_input(self, dataloader):
        """Unpack input data and perform data allocation steps

        Parameters:

            dataloader (dict): include data itself and metadata information

        """

        self.X = dataloader["X"]

        self.Y = dataloader["Y"]

        self.ids = dataloader["IDS"]


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        
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


    def forward(self):

        #create latent input 
        latent_input = torch.randn(self.batch_size, self.max_prot_len, self.latent_dim)

        #create generated sequences
        self.gen_seq = self.net_g(latent_input, self.X)

        #find predicted objectives
        r_x = F.gumbel_softmax(gen_seq,tau=1,hard=True)
        self.x_hat = self.net_r(r_x)

        #calculate discriminator outputs
        self.disc_output_real = self.net_d(self.Y.long(), self.X)
        self.disc_output_fake = self.net_d(self.gen_seq.long(), self.X)


    def backward_seq(self):
        #calculate loss of sequence from gan to label seq
        self.loss_seq = self.seq_loss_fn(self.gen_seq, self.Y) #send backwards
        self.loss_seq.backward()


    def backward_obj(self):
        self.loss_obj = self.obj_loss_fn(self.x_hat, self.X) #send backwards
        self.loss_obj.backward()


    def backward_d(self):
        loss_d_real = self.d_loss_fn(disc_output_real, self.real_labels)
        loss_d_fake = self.d_loss_fn(self.disc_output_fake, self.fake_labels)
        self.loss_d = loss_d_real + loss_d_fake #optimizable
        self.loss_d.backward()


    def backward_g(self):
        #calculate gan losses
        self.loss_gan = self.gan_loss_fn(self.disc_output_fake, self.real_labels)
        self.loss_g = self.loss_seq + self.loss_obj + self.loss_gan #optimizable
        self.loss_g.backward()

    
    def optimize_parameters(self):
        #forward pass
        self.forward()

        #optimize generator
        self.set_requires_grad([self.net_r],False)
        self.set_requires_grad([self.net_d],False)
        self.set_requires_grad([self.net_g],True)
        
        self.optimizer_g.zero_grad()

        self.backward_seq()
        self.backward_obj()
        self.set_requires_grad([self.net_r],False)
        self.backward_g()

        self.set_requires_grad([self.net_r],False)
        self.optimizer_g.step()

        #optimize discriminator
        self.set_requires_grad([self.net_d],True)

        self.optimizer_d.zero_grad()

        self.backward_d()

        self.optimizer_d.step()


    def save_params(self, epoch, iter, save_dir):
        """
        
        Parameters
        
            

        """

        #create file suffixes
        suffix_g = f"g_epoch_{epoch}_iters_{iters}"
        suffix_d = f"d_epoch_{epoch}_iters_{iters}"

        #get save directories
        save_dir_g = os.path.join(save_dir, suffix_g)
        save_dir_d = os.path.join(save_dir, suffix_d)

        #save models
        torch.save(self.net_g.state_dict(), save_dir_g)
        torch.save(self.net_d.state_dict(), save_dir_d)


    def compute_visuals(self, epoch, iter, writer, map):
        """
        
        Parameters
        
            
            
        """

        with torch.no_grad(): #run in no grad to make sure it doesn't affect gradients

            #detach onehot vector
            output = self.r_x.detach()

            sequences = []

            #decode tokenized sequence
            for tensor in 
            

                #map from onehot to sequence

                #treat 0-based onehot vectors as pad


            #run ESMFold


            #get image saveable thing

            #convert to numpy and 




