import torch
import torch.nn as nn
import torch.nn.functional as F 

#define a generator class with an RNN-based network
class RNN_generator(nn.Module):
    def __init__(self, latent_dim, objective_dim, hidden_dim, num_rnn_layers, output_dim):
        """
        Parameters

            latent_dim (int): last dimension of input latent random noise vector

            objective_dim (int): last dimension of input objective tensor

            hidden_dim (int): dimensionality of RNN hidden layers

            num_rnn_layers (int): number of RNN modules in model

            output_dim (int): number of output categories

        """
        
        super(RNN_generator, self).__init__()

        self.rnn = nn.RNN(latent_dim + objective_dim, hidden_dim, num_rnn_layers, batch_first=True)

        self.activation1 = nn.LeakyReLU(0.2)

        self.fc1 = nn.Linear(hidden_dim,output_dim)

        self.activation2 = nn.Softmax(1)
    

    def forward(self, latent_noise, objectives):
        x = torch.concat([latent_noise, objectives], dim=-1)

        x, _ = self.rnn(x)

        x = self.fc1(x)

        out = self.activation2(x)

        return out




















#Define discriminator
class Discriminator(nn.Module):
    def __init__(self, layer_sizes, objective_dim, seq_len, num_classes=21):
        """
        Parameters
            layer_sizes (list) - sizes of the linear layers in the model

            objective_dim (int) - size of the input objective vector

            seq_len (int) - length of the sequences,

            num_classes (int) - number of AA outputs in each sequence
            
        """

        super(Discriminator, self).__init__()

        self.layer_sizes = layer_sizes
        self.objective_dim = objective_dim
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.objective_embedding = 0
