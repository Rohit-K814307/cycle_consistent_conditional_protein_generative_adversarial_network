#package-specific imports
import gan_protein_structural_requirements.utils as utils
from gan_protein_structural_requirements.utils.protein_visualizer import jupy_viz_obj as jvo, jupy_viz_file as jvf
from gan_protein_structural_requirements.utils.folding_models import esm_batch_predict as esm_batch_predict, esm_predict as esm_predict, load_tokenizer as load_tokenizer
from gan_protein_structural_requirements.utils.extract_structure import extract_structures as extract_structures, untokenize as untokenize, get_untokenizer as get_untokenizer
from gan_protein_structural_requirements.data import class_ccc_progan_dataset as cdt
import gan_protein_structural_requirements.utils.folding_models as fold
from gan_protein_structural_requirements.models import networks
from gan_protein_structural_requirements.models.seqtovec_model import SeqToVecModel
from gan_protein_structural_requirements.train import train
from gan_protein_structural_requirements.test import test_seqtovec
from gan_protein_structural_requirements.data.class_ccc_progan_eval_dataset import Eval_Protein_dataset as edt
from gan_protein_structural_requirements.models.ccc_progan_model import RNN_CCCProGANModel

#remaining imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt


MAX_PROT_LEN = 1024
dtst = torch.load("gan_protein_structural_requirements/data/save/train.pt")
eval_dtst = torch.load("gan_protein_structural_requirements/data/save/test.pt")

####################################################

#train CCC-ProGAN with RNN-based generator

#setup values
objective_dim = 9
sequence_dim = 20
max_prot_len = 1024
device = torch.device('cpu')
root_save_dir = "/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements"
path_to_r = "/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/checkpoints/seqtovec/epoch_100_iters_1101"
path_to_esm = "/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/utils/esmfoldv1"
model_name = "rnn_ccc_progan"
save_freq = 10


#hparams
LR = [1e-3, 5e-3]
LR_DECAYS = [0.95, 0.95]
LR_BETA = [0.9,0.9]
EPSILON = [1e-8,1e-8]
WEIGHT_DECAY = [1e-3,1e-3]
BATCH_SIZE = 12
EPOCHS = 100
HIDDEN_DIM = 128
LATENT_DIM = 100
NUM_RNN_LAYERS = 3
LAMBDA_SEQ = 1
LAMBDA_OBJ = 1
LAMBDA_G = 1
LAMBDA_D = 1
COSINE_SIMILARITY_EPSILON = 1e-3

#create a dataset
dataloader = DataLoader(dataset=dtst,
                   batch_size=BATCH_SIZE,
                   shuffle=True)


#load model
rnn_ccc_progan = RNN_CCCProGANModel(latent_dim=LATENT_DIM,
                                    objective_dim=objective_dim,
                                    hidden_dim=HIDDEN_DIM,
                                    num_rnn_layers=NUM_RNN_LAYERS,
                                    sequence_dim=sequence_dim,
                                    batch_size=BATCH_SIZE,
                                    max_prot_len=MAX_PROT_LEN,
                                    lr=LR,
                                    lr_beta=LR_BETA,
                                    epsilon=EPSILON,
                                    lambda_seq=LAMBDA_SEQ,
                                    lambda_obj=LAMBDA_OBJ,
                                    lambda_g=LAMBDA_G,
                                    lambda_d=LAMBDA_D,
                                    weight_decay=WEIGHT_DECAY,
                                    cos_eps=COSINE_SIMILARITY_EPSILON,
                                    path_to_r=path_to_r,
                                    path_to_esm=path_to_esm,
                                    viz=False)

train(model=rnn_ccc_progan,
      dataset=dataloader,
      epochs=EPOCHS,
      lr_decays=LR_DECAYS,
      device=device,
      save_freq=save_freq,
      root_dir=root_save_dir,
      model_name="rnn_ccc_progan",
      map=dtst.decode_cats,
      viz=False)