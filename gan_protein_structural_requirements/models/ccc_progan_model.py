from gan_protein_structural_requirements.models import networks
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import gan_protein_structural_requirements.utils.folding_models as fold
from gan_protein_structural_requirements.utils.folding_models import load_esm, convert_outputs_to_pdb
from gan_protein_structural_requirements.models import metrics as m
import gan_protein_structural_requirements.utils.protein_visualizer as protein_visualizer
from random import randint

class RNN_CCCProGANModel(nn.Module):
    def __init__(self, latent_dim, objective_dim, hidden_dim, num_rnn_layers, sequence_dim, batch_size, max_prot_len, lr, lr_beta, epsilon, lambda_seq, lambda_obj, lambda_g, lambda_d, weight_decay, cos_eps, path_to_r, path_to_esm, viz=True):
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

        self.net_g = networks.RNN_generator(latent_dim,objective_dim,hidden_dim,num_rnn_layers,sequence_dim).train()
        
        self.net_d = networks.Discriminator(batch_size, max_prot_len, [64, 32],objective_dim,sequence_dim).train()
        
        self.net_r = networks.SeqToVecEnsemble(sequence_dim, max_prot_len)
        self.net_r.load_state_dict(torch.load(path_to_r))
        self.net_r.eval()
        self.set_requires_grad([self.net_r],False)
        
        if viz:
            self.net_viz, self.viz_tokenizer = load_esm(path_to_esm, False)
            self.net_viz.eval()
            self.visual_names = ["proteins_TER"]

        self.score_names = ["seq_feasibility", "seq_diversity"]
        self.loss_names = ["loss_seq", "loss_obj", "loss_gan", "loss_g", "loss_d"]
        self.networks = ["net_g","net_d","net_r"]

        self.seq_loss_fn = nn.CosineSimilarity(dim=-1, eps=cos_eps) #https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1175&context=computerscidiss
        self.obj_loss_fn = nn.MSELoss()
        self.gan_loss_fn = nn.BCELoss()

        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),lr=lr[0], betas=(lr_beta[0], 0.999), eps=epsilon[0], weight_decay=weight_decay[0])
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),lr=lr[1], betas=(lr_beta[0], 0.999), eps=epsilon[0], weight_decay=weight_decay[1])
        self.optimizers = [self.optimizer_g, self.optimizer_d]

    def set_input(self, dataloader):
        """Unpack input data and perform data allocation steps

        Parameters:

            dataloader (dict): include data itself and metadata information

        """

        self.X = dataloader["X"]

        self.Y = dataloader["Y"]

        self.ids = dataloader["IDS"]

        self.real_labels = torch.ones(self.X.size(0), 1)
        self.fake_labels = torch.zeros(self.X.size(0), 1)


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


    def forward(self):

        #create latent input 
        latent_input = torch.randn(self.X.size(0), self.max_prot_len, self.latent_dim)

        #create generated sequences
        self.gen_seq = self.net_g(latent_input, self.X)

        #find predicted objectives
        self.r_x = F.gumbel_softmax(self.gen_seq,tau=1,hard=True)
        self.sec, self.pol = self.net_r(self.r_x.float().permute(0,2,1))
        self.x_hat = torch.cat([self.sec, self.pol], dim=-1)

        #calculate discriminator outputs
        self.disc_output_real = self.net_d(self.Y.long(), self.X.detach())
        self.disc_output_fake = self.net_d(self.gen_seq.detach().long(), self.X.detach())

        #calculate score metrics
        with torch.no_grad():

            preds = self.r_x.detach().clone()

            self.seq_feasibility = m.seq_feasibility(preds, self.Y.detach().clone())

            self.seq_diversity = m.seq_diversity(preds)
        


    def backward_d(self):
        loss_d_real = self.gan_loss_fn(self.disc_output_real, self.real_labels) * self.lambda_d
        loss_d_fake = self.gan_loss_fn(self.disc_output_fake, self.fake_labels) * self.lambda_d

        self.loss_d = (loss_d_real + loss_d_fake) / 2 #optimizable
        self.loss_d.backward()


    def backward_g(self):
        #calculate gan losses
        self.loss_obj = self.obj_loss_fn(self.x_hat, self.X[:,0,:].float()) * self.lambda_obj
        self.loss_seq = self.seq_loss_fn(self.r_x, self.Y.float()).pow(2).mean() * self.lambda_seq #square of cos similarity to get rid of potential negatives since range of cos is [-1,1]
        self.loss_gan = self.gan_loss_fn(self.net_d(self.gen_seq.long(), self.X), self.real_labels) * self.lambda_g

        #summed entire gan loss
        self.loss_g = self.loss_seq + self.loss_obj + self.loss_gan #optimizable

        self.loss_g.backward()

    
    def optimize_parameters(self):
        #forward pass
        self.forward()

        #optimize generator
        self.set_requires_grad([self.net_r],False)
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        #optimize discriminator
        self.set_requires_grad([self.net_r],False)
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        


    def save_model(self, save_dir, epoch, iter):
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
            output = self.r_x.detach().clone()

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
                
            rand_idx = randint(0, len(sequences))

            tokenized_input = self.viz_tokenizer([sequences[rand_idx]], return_tensors="pt", add_special_tokens=False)['input_ids']

            _, pdb = fold.esm_predict(self.net_viz,tokenized_input)

            return pdb, self.ids[rand_idx]
    
    def get_viz(self, map):
        """
        Parameters
        
            map (dict) -- dictionary of mappings for onehot encoding indices
            
        """

        pdb, id = self.compute_pdb(map)

        view = protein_visualizer.jupy_viz_obj(pdb)

        plot = protein_visualizer.view_to_plt(view)

        return plot, id


