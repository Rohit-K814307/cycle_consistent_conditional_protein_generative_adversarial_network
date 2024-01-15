import torch


def seq_feasibility(preds, actuals):
    """

    Parameters:

        preds: predictions (onehot)

        actuals: actual y-hats

    """

    return ((preds == actuals).sum().item()) / preds.size(0)
    
def hamming_distance(chain1, chain2):
    return sum(c1 != c2 for c1, c2 in zip(chain1, chain2))


def seq_diversity(preds):
    """Finds average hamming distance of sequences in dataset
    
    Parameters:
    
        preds -- predictions in onehot encoded format
        
    """

    preds = torch.argmax(preds, dim=-1).numpy()

    hamming_sum = 0
    num_comparisons = 0

    for j in range(len(preds)):
        for k in range(j, len(preds)):
            hamming_sum += hamming_distance(preds[j],preds[k])
            num_comparisons += 1
    
    return hamming_sum/num_comparisons



    