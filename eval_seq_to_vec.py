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


sequence_dim = 20
max_prot_len = 1024
MAX_PROT_LEN = 1024
eval_dtst = torch.load("gan_protein_structural_requirements/data/save/test.pt")

print("|-----------------------Begin Evaluation-----------------------|")

#evaluate 
net = networks.SeqToVecEnsemble(sequence_dim, max_prot_len)
seqtovec_eval_metrics = test_seqtovec(
    test_dataset=eval_dtst,
    model=net,
    model_save_path="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/checkpoints/seqtovec/epoch_100_iters_1101")