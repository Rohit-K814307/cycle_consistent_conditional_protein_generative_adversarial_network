import torch


def seq_feasibility(preds, actuals):
    """

    Parameters:

        preds: predictions (onehot)

        actuals: actual y-hats

    """

    return ((preds == actuals).sum().item()) / preds.shape()[0].item()
    
def smith_waterman_similarity(tensor1, tensor2, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    m, n = len(tensor1), len(tensor2)

    dp_matrix = torch.zeros((m + 1, n + 1))

    # Fill the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp_matrix[i - 1, j - 1] + (match_score if tensor1[i - 1] == tensor2[j - 1] else mismatch_penalty)
            delete = dp_matrix[i - 1, j] + gap_penalty
            insert = dp_matrix[i, j - 1] + gap_penalty
            dp_matrix[i, j] = max(0, match, delete, insert)

    max_score = torch.max(dp_matrix)

    return max_score.item()


def seq_diversity(preds, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
    """
    
    Parameters:
    
        preds -- predictions in onehot encoded format
        
    """

    preds = torch.argmax(preds, dim=-1)

    n = preds.size(0)
    similarity_sums = torch.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity_score = smith_waterman_similarity(preds[i], preds[j], match_score, mismatch_penalty, gap_penalty)
            similarity_sums[i, j] = similarity_score
            similarity_sums[j, i] = similarity_score  # The matrix is symmetric

    #avg similarity
    total_similarity = torch.sum(similarity_sums)
    num_pairs = n * (n - 1) / 2  # Number of unique pairs
    average_similarity = total_similarity / num_pairs

    return average_similarity.item()