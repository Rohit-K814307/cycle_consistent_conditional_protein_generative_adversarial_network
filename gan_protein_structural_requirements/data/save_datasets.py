from gan_protein_structural_requirements.data import class_ccc_progan_dataset as cdt
from gan_protein_structural_requirements.data.class_ccc_progan_eval_dataset import Eval_Protein_dataset as edt
import torch

#collect dataset values
MIN_PROT_LEN = 200
MAX_PROT_LEN = 1024


train_dtst = cdt.Protein_dataset(
    root_dir="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/raw/train",
    min_prot_len=200,
    max_prot_len=1024
    )

eval_dtst = edt(
    root_dir="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/raw/test",
    min_prot_len=200,
    max_prot_len=1024)


torch.save(train_dtst,"/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/save/train.pt")
torch.save(eval_dtst,"/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/save/test.pt")