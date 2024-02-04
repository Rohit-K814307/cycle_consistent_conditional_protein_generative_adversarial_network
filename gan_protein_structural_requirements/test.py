import gan_protein_structural_requirements.models.metrics as m
from gan_protein_structural_requirements.models import random_sample_protein, process_outs
import gan_protein_structural_requirements.utils.protein_visualizer as viz
from gan_protein_structural_requirements.models.networks import SeqToVecEnsemble
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap
import json

def test_seqtovec(test_dataset, model, model_save_path=None, json_save_dir=None):
    """
    Parameters
        
        test_dataset: loaded dataset with test features and labels
        
        model: model instance to be tested
        
        model_save_path: save path of trained model
        
    """
    
    if model_save_path is not None:
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

    out_json = {"Total Loss": loss.item(),
            "Alpha Helix Loss": mse_ahelix.item(),
            "Beta Bridge Loss": mse_betabridge.item(),
            "Strand Loss": mse_strand.item(),
            "3-10 Helix Loss": mse_310helix.item(),
            "Pi Helix Loss": mse_pi.item(),
            "Turn Loss": mse_turn.item(),
            "Bend Loss":mse_bend.item(),
            "None Loss":mse_none.item(),
            "Polarity Loss":mse_pol.item()}
    
    if json_save_dir is not None:
        os.makedirs(json_save_dir, exist_ok=True)

        with open(f"{json_save_dir}/metrics.json", "w") as outfile:
            json.dump(out_json, outfile)
    else:
        return out_json

########################################################################################
#protein generation model evaluation and testing code

#define load generator function
def load_generator(model, save_path):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    return model


#randomly generate num_proteins proteins and get diversity of outputs and sequences
def get_metrics_rand(model, path_to_r, num_proteins, map, max_prot_len, latent_dim, vocab_size):

    with torch.no_grad():

        model_r = SeqToVecEnsemble(vocab_size, max_prot_len)
        model_r.load_state_dict(torch.load(path_to_r))
        model_r.eval()
        
        data, latent_input = random_sample_protein(num_proteins, latent_dim)
        
        outs = model(latent_input, data)
        onehot = F.gumbel_softmax(outs, tau=1, hard=True)
        sec, pol= model_r(onehot)
        x_hat = torch.cat([sec, pol], dim=-1)
        obj_loss_fn = nn.MSELoss()

    sequences = process_outs(outs.permute(0,2,1), map)

    pdbs = viz.esm_predict_api_batch(sequences)

    return {
        "PDBS":pdbs,
        "Design Objectives":data.numpy(),
        "Objective Loss":obj_loss_fn(x_hat, data).item(), 
        "Sequences":sequences
    }


#get metrics for one dataset of processed x and processed y
def get_metrics_dtst(model, path_to_r, X, y, max_prot_len, latent_dim, vocab_size, map):
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

        latent_input = torch.randn(X.size(0), latent_dim)
        out = model(latent_input, X[:,0,:])
        onehot = F.gumbel_softmax(out,tau=1,hard=True)
        seq_loss_fn = nn.CrossEntropyLoss(ignore_index=21)

        model_r = SeqToVecEnsemble(vocab_size, max_prot_len)
        model_r.eval()
        model_r.load_state_dict(torch.load(path_to_r))
        sec, pol= model_r(onehot)
        x_hat = torch.cat([sec, pol], dim=-1)
        obj_loss_fn = nn.MSELoss()

        seq_loss = seq_loss_fn(out, torch.argmax(y.float(),dim=-1)).item()
        obj_loss = obj_loss_fn(x_hat, X[:,0,:].float()).item()

    sequences_pred = process_outs(out.permute(0,2,1), map)[0:15]
    sequences_labels = process_outs(y, map, gumbel=False)[0:15]

    pdbs_hat = viz.esm_predict_api_batch(sequences_pred)
    pdbs_labels = viz.esm_predict_api_batch(sequences_labels)

    rmsd, scores = m.avg_rmsd(pdbs_hat, pdbs_labels)

    metrics = {
        "Sequence Loss": seq_loss,
        "Objective Loss": obj_loss,
        "Average RMSD": rmsd,
        "Predicted Sequences": sequences_pred,
        "Label Sequences": sequences_labels,
        "Label PDBS": pdbs_labels,
        "Predicted PDBS": pdbs_hat,
        "Label Objectives": X[0:15,0,:].numpy(),
        "Predicted Objectives": x_hat[0:15].numpy(),
        "Prot RMSDs": scores
    }

    return metrics
    

def create_captioned_image(pdb, objectives):
    image = viz.viz_protein_seq(pdb)

    caption = ""
    for objective in objectives:
        caption += str(objective) + ", "
    caption = "[" + caption[0:len(caption)-1] + "]"

    wrapped_caption = textwrap.fill(caption, width=70) 

    fig, ax = plt.subplots()
    ax.imshow(np.array(image))
    ax.set_title(wrapped_caption,fontdict={'fontsize':12,'horizontalalignment':'center'})
    ax.axis('off')
    plt.close()
    
    return fig


def evaluate_generator(path_to_G, instance_G, path_to_R, dataset, max_prot_len, latent_dim, vocab_size, num_proteins, results_save_root_dir=None):

    print("|----------------------EVALUATION---------------------------|")
    model_g = load_generator(instance_G,path_to_G)

    #get metrics data
    metrics_test_data = get_metrics_dtst(model_g, path_to_R, dataset.X, dataset.Y, max_prot_len, latent_dim, vocab_size, dataset.decode_cats)
    metrics_random_data = get_metrics_rand(model_g, path_to_R, num_proteins, dataset.decode_cats, max_prot_len, latent_dim, vocab_size)

    #get visual data from random proteins
    pdbs = metrics_random_data["PDBS"]

    images_rand = []
    #create matplotlib image graphs with objectives as captions for random dataset
    for i in range(len(pdbs)):
        images_rand.append(create_captioned_image(pdbs[i], metrics_random_data["Design Objectives"][i]))
    
    #create matplotlib image graphs with objectives as captions for predicted and label pdbs and objectives
    images_pred_test_data = []
    for i in range(len(metrics_test_data["Predicted PDBS"])):
        images_pred_test_data.append(
            create_captioned_image(metrics_test_data["Predicted PDBS"][i],
                                   metrics_test_data["Predicted Objectives"][i]
            )
        )

    images_label_test_data = []
    for i in range(len(metrics_test_data["Label PDBS"])):
        images_label_test_data.append(
            create_captioned_image(metrics_test_data["Label PDBS"][i],
                                   metrics_test_data["Label Objectives"][i]
            )
        )

    print()
    print()
    print("|-------------------Test Dataset Metrics--------------------|")
    for key in metrics_test_data.keys():
        if key != "Predicted Sequences" and key != "Label Sequences" and key != "Label PDBS" and key != "Predicted PDBS" and key != "Label Objectives" and key != "Predicted Objectives" and key != "Prot RMSDs":
            print(str(key) + ": " + str(metrics_test_data[key]))
            print()
    print()
    print()
    
    print("|-------------------Random Input Metrics--------------------|")
    for key in metrics_random_data.keys():
        if key != "Sequences" and key != "PDBS" and key != "Design Objectives":
            print(str(key) + ": " + str(metrics_random_data[key]))
            print()

    if results_save_root_dir is not None:
        print(f"|--------------Saving results to {results_save_root_dir}---------------|")

        #save predicted and label dirs with images 
        image_dir = os.path.join(results_save_root_dir,"images")
        os.makedirs(image_dir,exist_ok=True)

        image_paths_rand = [os.path.join(image_dir, "random", f"image_{i}.png") for i in range(len(images_rand))]
        image_paths_test_labels = [os.path.join(image_dir, "test", "label" , f"image_{i}.png") for i in range(len(images_label_test_data))]
        image_paths_test_preds = [os.path.join(image_dir, "test", "pred" , f"image_{i}.png") for i in range(len(images_pred_test_data))]
        
        metrics_random_data["Image Paths"] = image_paths_rand
        metrics_test_data["Label Image Paths"] = image_paths_test_labels
        metrics_test_data["Pred Image Paths"] = image_paths_test_preds

        os.makedirs(os.path.join(image_dir, "random"), exist_ok=True)
        os.makedirs(os.path.join(image_dir, "test", "label"), exist_ok=True)
        os.makedirs(os.path.join(image_dir, "test", "pred"), exist_ok=True)

        for i in range(len(images_rand)):
            images_rand[i].savefig(image_paths_rand[i],transparent=True,format="png")


        for i in range(len(image_paths_test_labels)):
            images_label_test_data[i].savefig(image_paths_test_labels[i], transparent=True, format="png")
        
        for i in range(len(image_paths_test_preds)):
            images_pred_test_data[i].savefig(image_paths_test_preds[i], transparent=True, format="png")




        #for blasting results
        
        string_list = metrics_random_data["Sequences"]
        max_strings_per_file=5
        file_counter=1


        for i in range(0, len(string_list), max_strings_per_file):

            with open(os.path.join(results_save_root_dir,f'output_{file_counter}.txt'), 'w') as file:

                for string in string_list[i:i+max_strings_per_file]:
                    file.write(">\n")
                    file.write(f"{string}\n")

            file_counter += 1

        print("BLAST SEQUENCES FOR SCORES through https://www.uniprot.org/blast")

    metrics_test_data["Label Objectives"] = metrics_test_data["Label Objectives"].tolist()
    metrics_test_data["Predicted Objectives"] = metrics_test_data["Predicted Objectives"].tolist()
    metrics_random_data["Design Objectives"] = metrics_random_data["Design Objectives"].tolist()

    if results_save_root_dir is not None:
        json_save_dict = {"Test Data Metrics": metrics_test_data,
                          "Random Data Metrics": metrics_random_data}
        
        with open(f"{results_save_root_dir}/results.json", "w") as outfile:
            json.dump(json_save_dict, outfile)

    else:
        return metrics_test_data, metrics_random_data, images_label_test_data, images_pred_test_data, images_rand