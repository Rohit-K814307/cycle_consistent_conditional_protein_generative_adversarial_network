import torch
import torch.nn as nn
import torch.nn.functional as F 

#define a generator class with an RNN-based network
class RNN_generator(nn.Module):
    def __init__(self, latent_dim, objective_dim, hidden_size, num_rnn_layers, num_classes=21):
        """
        Parameters
        
        latent_dim (int) - arbitrary dimension the size of the latent input: can be anything

        objective_dim (int) - number of features of objective input

        hidden_size (int) - size of hidden layers

        num_rnn_layers (int) - number of RNN modules in model

        num_classes (int) - number of AA outputs in each sequence

        """

        super(RNN_generator,self).__init__()

        self.latent_dim = latent_dim
        self.objective_dim = objective_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_rnn_layers = num_rnn_layers

        self.latent_fc = nn.Linear(self.latent_dim + self.objective_dim, hidden_size)

        self.activation1 = nn.LeakyReLU(0.2)

        self.rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_rnn_layers)

        self.output_fc = nn.Linear(self.hidden_size,self.num_classes)

        self.activation2 = nn.Softmax(dim=-1)

    def forward(self, latent_input, objective_input):
        #prepare input with concatenation
        x = torch.cat([latent_input, objective_input], dim=-1)

        #perform NN operations
        x = self.latent_fc(x)

        x = self.activation1(x)

        x, _ = self.rnn(x)

        x = self.output_fc(x)

        out = self.activation2(x)

        return out
    


class Discriminator(nn.Module):
    def __init__(self, ):
        """
        Parameters
            
        """
        super(Discriminator, self).__init__()
    pass
