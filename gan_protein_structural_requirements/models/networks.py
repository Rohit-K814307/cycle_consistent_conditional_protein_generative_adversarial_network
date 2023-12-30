import torch
import torch.nn as nn
import torch.nn.functional as F 


#define a generator class with a RNN-based network

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


#Define discriminator model

class Discriminator(nn.Module):
    def __init__(self, batch_size, seq_length, discriminator_layer_size, objective_dim, num_classes=21, embed_size=32):
        """
        Parameters

            batch_size (int): batch size of data

            seq_length (int): length of output AA sequences

            discriminator_layer_size (list): list with size of discriminator layers

            objective_dim (int): last dimension of objective input

            num_classes (int): number of classes in onehot vector of sequence

            embed_size (int): embedding size for onehot sequence

        """

        super(Discriminator, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding = nn.Embedding(num_classes,embed_size)

        self.fc1 = nn.Linear(((num_classes * embed_size) + objective_dim) * seq_length, discriminator_layer_size[0])

        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1])

        self.leakyrelu2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(discriminator_layer_size[1], 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, objectives):
        
        seq = self.embedding(seq).view(self.batch_size, self.seq_length,-1)

        x = torch.cat([seq,objectives],dim=-1)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.leakyrelu1(x)

        x = self.fc2(x)

        x = self.leakyrelu2(x)

        x = self.fc3(x)

        out = self.sigmoid(x)

        return out
    

#define the full protein folding model for easier loss calculations

class ProteinFold(nn.Module):
    def __init__(self, ):
        """
        Parameters


        
        """

        super(ProteinFold,self).__init__()

    def forward(self, seq):
        
        #decode onehot sequence


        #tokenize for esm


        #pass through esm


        #get DSSP+polarity

        
        #create objective vector


        #repeat sequence_length times in objective vector