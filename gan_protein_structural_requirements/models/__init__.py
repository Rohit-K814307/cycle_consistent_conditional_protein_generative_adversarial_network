import os
import torch
import torch.nn.functional as F

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_metrics(model, score_names):
    """Get metrics dictionary
    
    Parameters
    
        model: model to sort through
        
        score_names: model.losses or model.scores
        
    """
    return {score_names[i] : getattr(model, score_names[i]) for i in range(len(score_names))}

def print_metrics(epoch, iter, scores,losses):
    print(f"|-------------------------Epoch {epoch}, iter {iter}-----------------------|")
    print_string = ""

    for _, key in enumerate(scores):
        print_string += f"{key}: {scores[key]}\n"
    
    for _, key in enumerate(losses):
        print_string += f"{key}: {losses[key]}\n"

    print(print_string)


def random_sample_protein(num_prot, latent_dim):

    sec = F.softmax(torch.randn((num_prot, 8)), dim=-1)
    pol = F.sigmoid(torch.randn((num_prot, 1)))

    return torch.cat([sec,pol],dim=-1), torch.randn((num_prot, latent_dim))


def process_outs(outs, map, gumbel=True):
    if gumbel:
        output = F.gumbel_softmax(outs,tau=1,hard=True)
    else:
        output = outs
    
    sequences = []

    for sequence in output:
        seq = ""
        for residue in sequence:
            seq += map[torch.argmax(residue,dim=-1).item()]

        seq = seq.replace("-","")

        sequences.append(seq)

    return sequences