import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import os
import gan_protein_structural_requirements.models as models

def train_seqtovec(model, dataset, epochs, lr_decay, device, save_freq, root_dir, model_name):
    """Train a model
    
    Parameters:

        model (nn.Module): model to be trained

        dataset (torch.utils.data.DataLoader): dataloader object with training data

        epochs (int): number of epochs to train model

        lr_decay (float): learning rate decay over training time (exponential)

        device (torch.device): device to run training on

        save_freq (int): number of epochs to save model in

        root_dir (string): directory where model checkpoints, logs, and information should be saved

        model_name (string): name of model

    """

    save_dir = os.path.join(root_dir, "checkpoints", model_name)
    log_dir = os.path.join(root_dir, "checkpoints", model_name, "logs")
    os.makedirs(save_dir)
    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    model.to(device)
    model.train()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model.optim, gamma=lr_decay)
    total_iters = 1
    model_metrics = model.score_names
    model_losses = model.loss_names

    print("|-----------------------initializing model-----------------------|")
    print()
    print(model)
    print()
    print("|-----------------------Beginning Training-----------------------|")

    model.train()
    for epoch in range(1, epochs+1):

        iter_nums = 0

        for iter, data in enumerate(dataset):

            #set model inputs and processing/unpacking
            model.set_input(data)

            #optimize model parameters
            model.optimize_parameters()

            #add tensorboard logs
            scores = models.get_metrics(model, model_metrics)
            losses = models.get_metrics(model, model_losses)

            writer.add_scalars("scores",scores,total_iters)
            writer.add_scalars("losses",losses,total_iters)

            #print metrics
            models.print_metrics(epoch,iter+1,scores,losses)
            print()
            print()

            iter_nums += 1

        #calculate total number of iterations
        total_iters += iter_nums + 1

        #save model
        if epoch % save_freq == 0:
            print(f"|----------Saving model at end of epoch {epoch}, iters {total_iters}.----------|")
            print()
            print()
            model.save_model(save_dir, epoch, total_iters)

        #decay learning rate
        lr_scheduler.step()

    writer.close()
    print("|-------------------------End of Training------------------------|")