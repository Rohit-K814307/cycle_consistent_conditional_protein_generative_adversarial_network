import gan_protein_structural_requirements.models as models
import gan_protein_structural_requirements.models.metrics as m
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import os
import numpy as np, numpy.random
import random

def test_seqtovec(test_dataset, model, model_save_path):
    """
    Parameters
        
        test_dataset: loaded dataset with test features and labels
        
        model: model instance to be tested
        
        model_save_path: save path of trained model
        
    """

    model.load_state_dict(torch.load(model_save_path))

    model.eval()

    loss_fn = nn.MSELoss()

    with torch.no_grad():

        data = test_dataset[:]

        y = data["X"].float()[:,0,:]

        X = data["Y"].float().permute(0,2,1)

        y_sec, y_pol = model(X)

        y_hat = torch.cat([y_sec, y_pol], dim=-1)

        loss = loss_fn(y_hat, y)

        mse_ahelix = loss_fn(y_hat[:,0],y[:,0])
        mse_betabridge = loss_fn(y_hat[:,1],y[:,1])
        mse_strand = loss_fn(y_hat[:,2],y[:,2])
        mse_310helix = loss_fn(y_hat[:,3],y[:,3])
        mse_pi = loss_fn(y_hat[:,4],y[:,4])
        mse_turn = loss_fn(y_hat[:,5],y[:,5])
        mse_bend = loss_fn(y_hat[:,6],y[:,6])
        mse_none = loss_fn(y_hat[:,7],y[:,7])
        mse_pol = loss_fn(y_hat[:,8],y[:,8])

        print(f"Total Loss: {loss}")

        print(f"Alpha Helix Loss: {mse_ahelix}")

        print(f"Beta Bridge Loss: {mse_betabridge}")

        print(f"Strand Loss: {mse_strand}")

        print(f"3-10 Helix Loss: {mse_310helix}")

        print(f"Pi Helix Loss: {mse_pi}")

        print(f"Turn Loss: {mse_turn}")

        print(f"Bend Loss: {mse_bend}")

        print(f"None Loss: {mse_none}")

        print(f"Polarity Loss: {mse_pol}")

    return {"Total Loss": loss,
            "Alpha Helix Loss": mse_ahelix,
            "Beta Bridge Loss": mse_betabridge,
            "Strand Loss": mse_strand,
            "3-10 Helix Loss": mse_310helix,
            "Pi Helix Loss": mse_pi,
            "Turn Loss": mse_turn,
            "Bend Loss":mse_bend,
            "None Loss":mse_none,
            "Polarity Loss":mse_pol}

########################################################################################
#protein generation model evaluation and testing code

#define load generator function
def load_generator(model, save_path):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    return model


#define function for cleaning and setting data for input to generator and regular inference
def set_data(input_batch, max_prot_len):
    """
    Parameters:
    
        input_batch (list): list of dictionaries in c8 DSSP + polarity format 
            with values of proportions
            dictionary keys:
                - "a_helix"
                - "beta-bridge"
                - "strand"
                - "3-10-helix"
                - "pi-helix"
                - "turn"
                - "bend"
                - "none"
                - "pol"

        max_prot_len
                
    """

    data = [[c["a_helix"], c["beta-bridge"], c["strand"], 
            c["3-10-helix"], c["pi-helix"], c["turn"],
            c["bend"], c["none"], c["pol"]] for c in input_batch]
    data = torch.tensor(data).unsqueeze(1).repeat(1,max_prot_len,1)

    return data


#create random input dataset of conditions and return full dataset
def create_rand_inputs(size, max_prot_len):

    data = []
    for _ in range(size):

        random_c8 = np.random.dirichlet(np.ones(8),size=1)[0]

        data.append({"a-helix":random_c8[0], "beta-bridge":random_c8[1], 
                      "strand":random_c8[2], "3-10-helix":random_c8[3], 
                      "pi-helix":random_c8[4], "turn":random_c8[5],
                      "bend":random_c8[6], "none":random_c8[7], 
                      "pol":random.randint(0,1)})
        
    return set_data(data, max_prot_len)


#define method to process outputs of model into onehot vectors
# and into string-based sequences of amino acids
def process_outs(outs, map):
    output = F.gumbel_softmax(outs,tau=1,hard=True)
    
    sequences = []

    for sequence in output:
        seq = ""
        for residue in sequence:
            seq += map[torch.argmax(residue,dim=-1).item()]

        sequences.append(seq)

    return sequences


#randomly generate num_proteins proteins and get diversity of outputs and sequences
def get_metrics_rand(model, num_proteins, map, max_prot_len, latent_dim):
    
    data = create_rand_inputs(num_proteins,max_prot_len)

    latent_input = torch.randn(data.size(0), max_prot_len, latent_dim)
    outs = model(latent_input, data)
    sequences = process_outs(outs, map)

    return {
        "Design Objectives":data[:,0,:].numpy(),
        "Sequences":sequences,
        "Diversity":m.seq_diversity(F.gumbel_softmax(outs, tau=1,hard=True))
    }


#get metrics for one dataset of processed x and processed y
def get_metrics_dtst(model, X, y, cos_eps, max_prot_len, latent_dim):
    """
    Parameters:
    
        model: model to train on
        
        X: input of dataset (should be processed)
        
        y: labels of dataset (should be processed)
        
        cos_eps: same epsilon value for cosine similarity used in training

        max_prot_len: max length of generated proteins

        latent_dim: same dimension of latent input value as used in training

    """

    with torch.no_grad():

        latent_input = torch.randn(X.size(0), max_prot_len, latent_dim)
        onehot = F.gumbel_softmax(model(latent_input, X),tau=1,hard=True)
        seq_loss_fn = nn.CosineSimilarity(dim=-1,eps=cos_eps)



        metrics = {"avg_accuracy":m.seq_feasibility(onehot,y),
                   "avg_diversity":m.seq_diversity(onehot),
                   "avg_cos_similarity":seq_loss_fn(onehot, y)}

        return metrics
    

def comput_pdb_from_sequences(sequences):
    pass