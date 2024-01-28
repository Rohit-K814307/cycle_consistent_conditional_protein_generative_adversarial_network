from gan_protein_structural_requirements.models import networks
import sys, os
import torch
import itertools
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn


class SeqToVecModel(nn.Module):
    def __init__(self, input_size, sequence_length, lr, lr_beta, epsilon):
        """Initialize SeqToVec Model
        
        Parameters:
        
            input_size (int): input vocab size of model

            sequence_length (int): length of each sequence
            
            lr (float): learning rate of model
            
            lr_beta: learning rate beta of model

            epsilon: epsilon value of model

        """

        super(SeqToVecModel, self).__init__()

        self.score_names = ['mse_ahelix','mse_betabridge',
                            'mse_strand','mse_310helix',
                            'mse_pi','mse_turn',
                            'mse_bend','mse_none',
                            'mse_pol']
        self.loss_names = ["loss_net"]
        self.visual_names = []
        self.networks = ["net"]

        self.score_metric = nn.MSELoss()
        self.sec_loss = nn.MSELoss()
        self.pol_loss = nn.MSELoss()

        self.net = networks.SeqToVecEnsemble(input_size, sequence_length)
        self.net.apply(self.init_weights)

        self.optim = torch.optim.Adam(self.net.parameters(),lr=lr, betas=(lr_beta, 0.999), eps=epsilon, weight_decay=0)

        self.optimizers = [self.optim]

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)
        if isinstance(module, nn.RNN):
            module.weight.data.normal_(mean=0.0,std=1.0)

    def set_input(self, input):
        """Unpack input data and perform data allocation steps

        Parameters:

            input (dict): include data itself and metadata information

        """

        self.X = input["Y"].float().permute(0,2,1)

        self.y = input["X"][:,0,:].float()
        
        self.y_sec = input["X"][:,0,:8].float()

        self.y_pol = input["X"][:,0,-1].unsqueeze(1).float()

        self.ids = input["IDS"]


    def forward(self):
        """Forward the model"""

        #compute output tensor
        self.sec, self.pol = self.net(self.X)

        self.y_hat = torch.concat([self.sec,self.pol],dim=-1)

        #compute scores for later visualization
        with torch.no_grad():
            self.mse_ahelix = self.score_metric(self.y_hat.detach()[:,0],self.y.detach()[:,0])
            self.mse_betabridge = self.score_metric(self.y_hat.detach()[:,1],self.y.detach()[:,1])
            self.mse_strand = self.score_metric(self.y_hat.detach()[:,2],self.y.detach()[:,2])
            self.mse_310helix = self.score_metric(self.y_hat.detach()[:,3],self.y.detach()[:,3])
            self.mse_pi = self.score_metric(self.y_hat.detach()[:,4],self.y.detach()[:,4])
            self.mse_turn = self.score_metric(self.y_hat.detach()[:,5],self.y.detach()[:,5])
            self.mse_bend = self.score_metric(self.y_hat.detach()[:,6],self.y.detach()[:,6])
            self.mse_none = self.score_metric(self.y_hat.detach()[:,7],self.y.detach()[:,7])
            self.mse_pol = self.score_metric(self.y_hat.detach()[:,8],self.y.detach()[:,8])


    def backward(self):

        #find loss of the network
        self.loss_sec = self.sec_loss(self.sec, self.y_sec)
        self.loss_pol = self.pol_loss(self.pol, self.y_pol)
        self.loss_net = self.loss_sec + self.loss_pol

        #send loss backward to compute gradients
        self.loss_net.backward()


    def optimize_parameters(self):

        self.forward()

        self.optim.zero_grad()

        self.backward()

        self.optim.step()

    def save_model(self, save_dir, epoch, iters):

        suffix = f"epoch_{epoch}_iters_{iters}"

        save_dir = os.path.join(save_dir, suffix)

        torch.save(self.net.state_dict(), save_dir)