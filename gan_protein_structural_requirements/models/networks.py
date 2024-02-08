import torch
import torch.nn as nn
import torch.nn.functional as F 
from gan_protein_structural_requirements.utils import folding_models as fold
from gan_protein_structural_requirements.utils import extract_structure as struct


#Define CNN Generator
class CNNGenerator(nn.Module):
    def __init__(self, input_dim, objective_dim, sequence_length, vocab_size, hidden_dims):
        super(CNNGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.objective_dim = objective_dim
        self.seq_len = sequence_length
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(self.input_dim + self.objective_dim, self.seq_len * self.hidden_dims[0])

        self.bn1 = nn.BatchNorm1d(self.seq_len * self.hidden_dims[0])
        self.deconv1 = nn.ConvTranspose1d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=1, stride=1, bias=False)
        self.act1 = nn.LeakyReLU()

        self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
        self.deconv2 = nn.ConvTranspose1d(self.hidden_dims[1], self.vocab_size, kernel_size=1, stride=1, bias=False)
        self.act2 = nn.LeakyReLU()


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform(module.weight)


    def forward(self, latent, objectives):

        x = torch.cat([objectives,latent],dim=-1)

        x = self.fc1(x)

        x = self.bn1(x)

        x = x.view(-1, self.hidden_dims[0], self.seq_len)

        x = self.deconv1(x)
        x = self.act1(x)

        x = self.bn2(x)
        x = self.deconv2(x)
        out = self.act2(x)

        return out


#Define basic discriminator model
class Discriminator(nn.Module):
    def __init__(self, seq_length, discriminator_layer_size, objective_dim, num_classes=22):
        """
        Parameters

            seq_length (int): length of output AA sequences

            discriminator_layer_size (list): list with size of discriminator layers

            objective_dim (int): last dimension of objective input

            num_classes (int): number of classes in onehot vector of sequence

        """

        super(Discriminator, self).__init__()

        self.nnout = nn.Sequential(
            nn.Linear(num_classes * seq_length + objective_dim * seq_length , discriminator_layer_size[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(discriminator_layer_size[2], discriminator_layer_size[3]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(discriminator_layer_size[3], discriminator_layer_size[4]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(discriminator_layer_size[4], 1),

            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.01)

    def forward(self, seq, objectives):

        x = torch.cat([seq,objectives],dim=1)

        x = x.view(x.size(0), -1)

        x = x + (0.1**0.5)*torch.randn(x.size(0),x.size(1))

        out = self.nnout(x)

        return out

###############################################
#Build architectures and related models for transformer-based Unet

class MLPBlock(nn.Module):
    def __init__(self, input_features, out_features, kernel_size, stride):
        super(MLPBlock, self).__init__()

        self.conv = nn.Conv1d(input_features, out_features, kernel_size=kernel_size, stride=stride)

        self.bn = nn.BatchNorm1d(out_features)

        self.act = nn.ReLU()

        self.apply(self.init_weights)


    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        out = self.act(x)

        return out

    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class TransformerBlock(nn.Module):
    def __init__(self, input_features, num_channels, kernel_size, stride, num_heads):
        super(TransformerBlock, self).__init__()

        self.fc1 = MLPBlock(num_channels, num_channels, kernel_size, stride)
        self.attention1 = nn.MultiheadAttention(embed_dim=input_features, num_heads=num_heads, batch_first=True)

        self.fc2 = MLPBlock(num_channels, num_channels, kernel_size, stride)
        self.attention2 = nn.MultiheadAttention(embed_dim=input_features, num_heads=num_heads, batch_first=True)

        self.fc3 = MLPBlock(num_channels, num_channels, kernel_size, stride)
        self.attention3 = nn.MultiheadAttention(embed_dim=input_features, num_heads=num_heads, batch_first=True)

        self.fc4 = MLPBlock(num_channels, num_channels, kernel_size, stride)
        self.attention4 = nn.MultiheadAttention(embed_dim=input_features, num_heads=num_heads, batch_first=True)

        self.fc5 = MLPBlock(num_channels, num_channels, kernel_size, stride)

        self.apply(self.init_weights)

    def forward(self, x):

        x = x + self.fc1(x)
        att , _ = self.attention1(x,x,x)
        x = x +  att

        x = x + self.fc2(x)
        att , _ = self.attention2(x,x,x)
        x = x +  att

        x = x + self.fc3(x)
        att , _ = self.attention3(x,x,x)
        x = x +  att

        x = x + self.fc4(x)
        att , _ = self.attention4(x,x,x)
        x = x +  att

        out = x + self.fc5(x)

        return out

    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, objective_dim, sequence_length, vocab_size, hidden_dims, kernel, stride):
        super(DecoderBlock, self).__init__()

        self.input_dim = input_dim
        self.objective_dim = objective_dim
        self.hidden_dims = hidden_dims
        self.seq_len = sequence_length
        self.vocab_size = vocab_size
        self.kernel = kernel
        self.stride = stride

        self.fc = nn.Linear(self.input_dim + self.objective_dim, self.seq_len * self.hidden_dims[0])

        self.conv1 = nn.Conv1d(self.hidden_dims[0], self.hidden_dims[1], self.kernel, stride)

        self.conv2 = nn.Conv1d(self.hidden_dims[1], self.hidden_dims[2], self.kernel, self.stride)

        self.conv3 = nn.Conv1d(self.hidden_dims[2], self.hidden_dims[3], self.kernel, self.stride)

        self.conv4 = nn.Conv1d(self.hidden_dims[3], self.hidden_dims[4], self.kernel, self.stride)

        self.trans1 = TransformerBlock(self.seq_len, self.hidden_dims[4],kernel, stride, 2)

        self.trans2 = TransformerBlock(self.seq_len, self.hidden_dims[4],kernel, stride, 2)

        self.ups1 = nn.ConvTranspose1d(self.hidden_dims[4], self.hidden_dims[4] * 2, kernel_size=1)

        self.trans3 = TransformerBlock(self.seq_len, self.hidden_dims[4],kernel, stride, 2)

        self.ups2 = nn.ConvTranspose1d(self.hidden_dims[4], self.hidden_dims[4] * 4, kernel_size=1)

        self.apply(self.init_weights)

    def forward(self, latent, objectives):

        x = torch.cat([objectives, latent], dim=-1)

        x = self.fc(x)

        x = x.view(-1, self.hidden_dims[0], self.seq_len)

        x1 = self.conv1(x)

        x2 = self.conv2(x1)

        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x5 = self.trans1(x4)

        x6 = self.trans2(x5)

        x7 = self.trans3(x6)

        x8 = self.ups1(x6)

        x9 = self.ups2(x7)

        out = torch.cat([x9, x8, x5], dim=1)

        return x1, x2, x3, x4, out
    
    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


class ConvBasicBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel_size):
        super(ConvBasicBlock, self).__init__()

        self.conv = nn.Conv1d(input_features, output_features, kernel_size=kernel_size)

        self.bn = nn.BatchNorm1d(output_features)

        self.act = nn.ReLU()

        self.apply(self.init_weights)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        out = self.act(x)

        return out
    
    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


class TransformerGenerator(nn.Module):
    def __init__(self, input_dim, objective_dim, sequence_length, vocab_size, hidden_dims):
        super(TransformerGenerator, self).__init__()

        self.input_dim = input_dim
        self.objective_dim = objective_dim
        self.hidden_dims = hidden_dims
        self.seq_len = sequence_length
        self.vocab_size = vocab_size

        #contractive path
        self.cont = DecoderBlock(self.input_dim, self.objective_dim, self.seq_len, self.vocab_size, self.hidden_dims[0], 1, 1)

        #expanding path
        self.conv1 = ConvBasicBlock(self.hidden_dims[0][4] + (self.hidden_dims[0][4] + self.hidden_dims[0][4] * 2 + self.hidden_dims[0][4] * 4), self.hidden_dims[1][0], 1)
        self.ups1 = nn.ConvTranspose1d(self.hidden_dims[1][0], self.hidden_dims[1][1], kernel_size=1)

        self.conv2 = ConvBasicBlock(self.hidden_dims[1][1] + self.hidden_dims[0][3], self.hidden_dims[1][2], 1)
        self.ups2 = nn.ConvTranspose1d(self.hidden_dims[1][2], self.hidden_dims[1][3], kernel_size=1)

        self.conv3 = ConvBasicBlock(self.hidden_dims[1][3] + self.hidden_dims[0][2], self.hidden_dims[1][4], 1)
        self.ups3 = nn.ConvTranspose1d(self.hidden_dims[1][4], self.hidden_dims[1][5], kernel_size=1)

        self.conv4 = ConvBasicBlock(self.hidden_dims[1][5] + self.hidden_dims[0][1], self.hidden_dims[1][6], 1)
        self.ups4 = nn.ConvTranspose1d(self.hidden_dims[1][6], self.hidden_dims[1][7], kernel_size=1)

        self.conv5 = ConvBasicBlock(self.hidden_dims[1][7], self.hidden_dims[1][8], 1)

        self.conv6 = nn.Conv1d(self.hidden_dims[1][8], self.hidden_dims[1][9], 1)
        self.ups5 = nn.ConvTranspose1d(self.hidden_dims[1][9], self.vocab_size, kernel_size=1)

        self.apply(self.init_weights)

    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, latent, objectives):

        x1, x2, x3, x4, x = self.cont(latent, objectives)

        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.ups1(x)

        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.ups2(x)

        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.ups3(x)

        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        x = self.ups4(x)

        x = self.conv5(x)

        x = self.conv6(x)
        out = self.ups5(x)

        return out


#Define convolutional discriminator model
class ConvolutionalDiscriminator(nn.Module):
    def __init__(self, hidden_dims, seq_len, objective_dim, num_classes=22):
        """
        Parameters

            hidden_dims (list): list with size of discriminator layers

            objective_dim (int): last dimension of objective input

            num_classes (int): number of classes in onehot vector of sequence

        """

        super(ConvolutionalDiscriminator, self).__init__()

        self.conv1 = nn.Conv1d(objective_dim + num_classes, hidden_dims[0], 1, 1)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], 1, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.act2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], 1, 1)
        self.bn2 = nn.BatchNorm1d(hidden_dims[2])
        self.act3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv1d(hidden_dims[2], hidden_dims[3], 1, 1)
        self.bn3 = nn.BatchNorm1d(hidden_dims[3])
        self.act4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv1d(hidden_dims[3], hidden_dims[4], 1, 1)
        self.bn4 = nn.BatchNorm1d(hidden_dims[4])
        self.act5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv1d(hidden_dims[4], hidden_dims[5], 1, 1)
        self.bn5 = nn.BatchNorm1d(hidden_dims[5])
        self.act6 = nn.LeakyReLU(0.2)

        self.flat = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc1 = nn.Linear(hidden_dims[5] * seq_len, hidden_dims[6])
        self.act7 = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dims[6], hidden_dims[7])
        self.act8 = nn.LeakyReLU(0.2)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dims[7], hidden_dims[8])
        self.act9 = nn.LeakyReLU(0.2)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_dims[8], hidden_dims[9])
        self.act10 = nn.LeakyReLU(0.2)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(hidden_dims[9], hidden_dims[10])
        self.act11 = nn.LeakyReLU(0.2)
        self.drop5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(hidden_dims[10], 1)

        self.act12 = nn.Sigmoid()

        self.apply(self.init_weights)

    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

        if isinstance(m, nn.ConvTranspose1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)

    def forward(self, seq, objectives):

        x = torch.cat([seq,objectives],dim=1)

        #add noise for better understanding
        x = self.conv1(x)
        x = self.act1(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.act3(x)


        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act4(x)
        
        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.act5(x)


        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.act6(x)

        x = self.flat(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.fc1(x)
        x = self.act7(x)
        #x = self.drop1(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.fc2(x)
        x = self.act8(x)
        #x = self.drop2(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.fc3(x)
        x = self.act9(x)
        #x = self.drop3(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.fc4(x)
        x = self.act10(x)
        #x = self.drop4(x)

        x = x + (0.1**0.5)*torch.randn_like(x)
        x = self.fc5(x)
        x = self.act11(x)
        #x = self.drop5(x)

        x = self.fc6(x)
        
        out = self.act12(x)


        return out


#define the frozen sequence to vector model
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

        self.conv1 = nn.Conv1d(input_size,layer_sizes[0], kernel_size=3, stride=1,padding=0)

        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(layer_sizes[0], layer_sizes[1], kernel_size=3, stride=1, padding=0)

        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv1d(layer_sizes[1], layer_sizes[2], kernel_size=3, stride=1, padding=0)

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

        self.conv1 = nn.Conv1d(input_size,layer_sizes[0], kernel_size=3, stride=1, padding=0)

        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(layer_sizes[0], layer_sizes[1], kernel_size=3, stride=1, padding=0)

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

    def forward(self, x):

        x = self.flatten(x)

        out = self.fc(x)

        return out

#Full ensemble model with Unet architecture
    
class SeqToSecondary(nn.Module):
    def __init__(self, input_size, sequence_length):
        """
        Parameters
        
            input_size (int): input size of model - should be the vocab size

            sequence_length (int): length of each sequence
            
        """

        super(SeqToSecondary, self).__init__()

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

        self.fc1 = nn.Linear(2168,64)

        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64,9)

        self.act2 = nn.ReLU()

        self.fc3 = PercentageRegressor(576,8)

        self.act3 = nn.Sigmoid()

    def forward(self, x):

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

        x1 = self.fc3(x1)

        out = self.act3(x1)

        return out
    

class SeqToVecEnsemble(nn.Module):
    def __init__(self, input_size, sequence_length):
        """
        Parameters
        
            input_size (int): input size of model - should be the vocab size

            sequence_length (int): length of each sequence
            
        """

        super(SeqToVecEnsemble, self).__init__()

        self.net_sec = SeqToSecondary(input_size, sequence_length)

        self.net_pol = PolarityRegressor(sequence_length, input_size, 64)

    def forward(self, x):

        sec = self.net_sec(x)

        pol = self.net_pol(x)

        return sec, pol
    
##########################################################################################
#create transformer-based networks