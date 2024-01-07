import os

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