import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import os
import torch.nn as nn
import gan_protein_structural_requirements.models as models

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

        y_hat = model(X, 1)

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