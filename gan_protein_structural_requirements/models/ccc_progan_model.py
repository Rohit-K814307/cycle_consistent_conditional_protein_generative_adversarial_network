from gan_protein_structural_requirements.models import networks
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from gan_protein_structural_requirements.utils.folding_models import load_esm, convert_outputs_to_pdb
from gan_protein_structural_requirements.models import metrics as m


class RNN_CCCProGANModel(nn.Module):
    def __init__(self, latent_dim, objective_dim, hidden_dim, num_rnn_layers, sequence_dim, batch_size, max_prot_len, lr, lr_beta, epsilon, path_to_r, path_to_esm, viz=True):
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

            path_to_r -- path to seqtovec (pretrained) model

            path_to_esm -- path to esmfold model

            viz -- load in esm models etc.
        
        """

        super(RNN_CCCProGANModel, self).__init__()

        self.batch_size = batch_size
        self.max_prot_len = max_prot_len
        self.latent_dim = latent_dim
        self.real_labels = torch.ones(self.batch_size, 1)
        self.fake_labels = torch.zeros(self.batch_size, 1)

        self.net_g = networks.RNN_generator(latent_dim,objective_dim,hidden_dim,num_rnn_layers,sequence_dim)
        self.net_d = networks.Discriminator(batch_size, max_prot_len, [64, 32],objective_dim,sequence_dim)
        self.net_r = networks.SeqToVecEnsemble(sequence_dim, max_prot_len).load_state_dict(torch.load(path_to_r))
        
        if viz:
            self.net_viz, self.viz_tokenizer = load_esm(path_to_esm, False)
            self.visual_names = []

        self.score_names = ["seq_feasibility", "seq_diversity"]
        self.loss_names = ["loss_seq", "loss_obj", "loss_gan", "loss_g", "loss_d"]
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
        r_x = F.gumbel_softmax(self.gen_seq,tau=1,hard=True)
        self.x_hat = self.net_r(r_x)

        #calculate discriminator outputs
        self.disc_output_real = self.net_d(self.Y.long(), self.X)
        self.disc_output_fake = self.net_d(self.gen_seq.long(), self.X)

        #calculate score metrics
        with torch.no_grad():

            preds = r_x.detach().copy()

            self.seq_feasibility = m.seq_feasibility(preds, self.Y.detach().copy())

            self.seq_diversity = m.seq_diversity(preds)





    def backward_seq(self):
        #calculate loss of sequence from gan to label seq
        self.loss_seq = self.seq_loss_fn(self.gen_seq, self.Y) #send backwards
        self.loss_seq.backward()


    def backward_obj(self):
        self.loss_obj = self.obj_loss_fn(self.x_hat, self.X) #send backwards
        self.loss_obj.backward()


    def backward_d(self):
        loss_d_real = self.d_loss_fn(self.disc_output_real, self.real_labels)
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
        
            epoch -- epoch of current training step

            iter -- iteration of current training step

            save_dir -- directory to save model to
            

        """

        #create file suffixes
        suffix_g = f"g_epoch_{epoch}_iters_{iter}"
        suffix_d = f"d_epoch_{epoch}_iters_{iter}"

        #get save directories
        save_dir_g = os.path.join(save_dir, suffix_g)
        save_dir_d = os.path.join(save_dir, suffix_d)

        #save models
        torch.save(self.net_g.state_dict(), save_dir_g)
        torch.save(self.net_d.state_dict(), save_dir_d)


    def compute_pdb(self, map):
        """
        
        Parameters
        
            map (dict) -- dictionary of mappings for onehot encoding indices
            
        """

        with torch.no_grad(): #run in no grad to make sure it doesn't affect gradients

            #detach onehot vector and extract actual sequences
            output = self.r_x.detach()

            sequences = []

            for sequence in output:
                seq = ""
                for residue in sequence:
                    if 1 in residue:
                        seq += map[torch.argmax(residue,dim=-1).item()]
                    else:
                        seq += "<pad>"

                sequences.append(seq)

            #run ESMFold and tokenizer

            tokenized_input = self.viz_tokenizer(sequences, return_tensors="pt", add_special_tokens=True)['input_ids']

            output_esm = self.net_viz(tokenized_input)

            pdb = convert_outputs_to_pdb(output_esm)

            return pdb