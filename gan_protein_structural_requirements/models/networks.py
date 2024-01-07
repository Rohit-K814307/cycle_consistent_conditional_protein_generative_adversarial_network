import torch
import torch.nn as nn
import torch.nn.functional as F 
from gan_protein_structural_requirements.utils import folding_models as fold
from gan_protein_structural_requirements.utils import extract_structure as struct


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

        ##################
        ##################
        #IMPORTANT:
        #MUST HAVE SOFTMAX
        #IN FULL ENSEMBLE
        #TO GET PROBABILI-
        #TIES
    

    def forward(self, latent_noise, objectives):
        x = torch.concat([latent_noise, objectives], dim=-1)

        x, _ = self.rnn(x)

        out = self.fc1(x)

        return out

#Define U-Net based generator




#Define LSTM based generator





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


#define the gradient locked sequence to vector model
#this model is a modified version of the ProteinUnet model by Kotowski et al
#https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.26432

#Basic block for Unet Model with 3 convolutional units
    
class BasicBlock3(nn.Module):
    def __init__(self, input_size, layer_sizes):
        """
        Parameters
        
            input_size (int): input size/last dim of model
            
            layer_sizes (list): length 3 list with the convolutional sizes of model

        """

        super(BasicBlock3,self).__init__()

        self.conv1 = nn.Conv1d(input_size,layer_sizes[0], kernel_size=7, stride=1,padding=0)

        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(layer_sizes[0], layer_sizes[1], kernel_size=7, stride=1, padding=0)

        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv1d(layer_sizes[1], layer_sizes[2], kernel_size=7, stride=1, padding=0)

        self.activation3 = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)

        x = self.activation1(x)

        x = self.conv2(x)

        x = self.activation2(x)

        x = self.conv3(x)

        out = self.activation3(x)

        return out
    
#Basic Block for Unet Model with 2 convolutional units
    
class BasicBlock2(nn.Module):
    def __init__(self, input_size, layer_sizes):
        """
        Parameters
        
            input_size (int): input size/last dim of model
            
            layer_sizes (list): length 2 list with the convolutional sizes of model
        """

        super(BasicBlock2, self).__init__()

        self.conv1 = nn.Conv1d(input_size,layer_sizes[0], kernel_size=7, stride=1, padding=0)

        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(layer_sizes[0], layer_sizes[1], kernel_size=7, stride=1, padding=0)

        self.activation2 = nn.ReLU()


    def forward(self, x):

        x = self.conv1(x)

        x = self.activation1(x)

        x = self.conv2(x)

        out = self.activation2(x)

        return out

#create model for polarity percentage prediction values

class PolarityRegressor(nn.Module):
    def __init__(self, sequence_length, vocab_size, hidden_size):
        """
        Parameters

            sequence_length (int): length of each sequence

            vocab_size (int): number of total one hot encodings

            hidden_size (int): hidden layer size for polarity
            
        """

        super(PolarityRegressor, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(sequence_length * vocab_size, hidden_size)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.flatten(x) 

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        out = self.sigmoid(x)

        return out


#Create end model for percentage prediction values

class PercentageRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Parameters
        
            input_size (int): input of size of model from previous ProteinUnet output
            
            output_size (int): output vector size of model - c8 SS or c3 SS
            
        """

        super(PercentageRegressor, self).__init__()

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(input_size,output_size)

        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.flatten(x)

        x = self.fc(x)

        out = self.act(x)

        return out

#Full ensemble model with Unet architecture
    
class SeqToVecEnsemble(nn.Module):
    def __init__(self, input_size, sequence_length):
        """
        Parameters
        
            input_size (int): input size of model - should be the vocab size

            sequence_length (int): length of each sequence
            
        """

        super(SeqToVecEnsemble, self).__init__()

        #Contractive path layers
        self.cont1 = BasicBlock3(input_size, [64,64,64])

        self.avgPool1 = nn.AvgPool1d(kernel_size=2)

        self.cont2 = BasicBlock3(64, [64,64,64])

        self.avgPool2 = nn.AvgPool1d(kernel_size=2)

        self.cont3 = BasicBlock3(64, [128,128,128])

        self.avgPool3 = nn.AvgPool1d(kernel_size=2)

        self.cont4 = BasicBlock3(128,[128,128,128])

        self.avgPool4 = nn.AvgPool1d(kernel_size=2)

        #Expanding path layers
        self.exp1 = BasicBlock2(128, [128,128])

        self.ups1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.exp2 = BasicBlock2(128,[128,128])

        self.ups2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.exp3 = BasicBlock2(128,[64,64])

        self.ups3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.exp4 = BasicBlock2(64,[64,64])

        self.ups4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.fc1 = nn.Linear(7640,64)

        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64,9)

        self.act2 = nn.ReLU()

        self.final1 = PercentageRegressor(576,8)

        self.final2 = PolarityRegressor(sequence_length, input_size, 64)

    def forward(self, x, temperature):

        x = F.gumbel_softmax(x,tau=temperature,hard=True)

        #compute operations for contracting path

        xc1 = self.cont1(x)

        xc1a = self.avgPool1(xc1)

        xc2 = self.cont2(xc1a)

        xc2a = self.avgPool2(xc2)

        xc3 = self.cont3(xc2a)

        xc3a = self.avgPool3(xc3)

        xc4 = self.cont4(xc3a)

        xc4a = self.avgPool4(xc4)


        #compute operations for expanding path

        xe1 = self.exp1(xc4a)

        xe1 = self.ups1(torch.concat([xe1,xc4], dim=-1))

        xe2 = self.exp2(xe1)

        xe2 = self.ups2(torch.concat([xe2,xc3], dim=-1))

        xe3 = self.exp3(xe2)

        xe3 = self.ups3(torch.concat([xe3,xc2], dim=-1))

        xe4 = self.exp4(xe3)

        xe4 = self.ups4(torch.concat([xe4,xc1], dim=-1))


        #compute operations final linear prediction layers

        x1 = self.fc1(xe4)

        x1 = self.act1(x1)

        x1 = self.fc2(x1)

        x1 = self.act2(x1)

        x1 = self.final1(x1)

        x2 = self.final2(x)

        out = torch.concat([x1, x2], dim=-1)

        return out