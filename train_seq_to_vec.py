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


#################################################################################################

#pretrain sequence to vector model
sequence_dim = 20
max_prot_len = 1024
device = torch.device('cpu')
root_save_dir = "/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/"
model_name = "seqtovec"

#hparams
LR = 5e-5
LR_DECAYS = [0.95]
LR_BETA = 0.9
EPSILON = 1e-08
BATCH_SIZE = 12
EPOCHS = 100

#make dataloader with dataset
dataloader = DataLoader(dataset=dtst,
                  batch_size=BATCH_SIZE,
                  shuffle=True)


seqtovec = SeqToVecModel(sequence_dim,max_prot_len,LR,LR_BETA,EPSILON)
train(model=seqtovec,
               dataset=dataloader,
               epochs=EPOCHS,
               lr_decays=LR_DECAYS,
               device=device,
               save_freq=5,
               root_dir=root_save_dir,
               model_name=model_name)

################################################################################################