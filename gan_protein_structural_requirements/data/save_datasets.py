from gan_protein_structural_requirements.data import class_ccc_progan_dataset as cdt
from gan_protein_structural_requirements.data import class_ccc_progan_eval_dataset as edt
import torch
import sys

#collect dataset values
MIN_PROT_LEN = int(sys.argv[1])
MAX_PROT_LEN = int(sys.argv[2])


train_dtst = cdt.Protein_dataset(
    root_dir="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/raw/model1/train",
    min_prot_len=MIN_PROT_LEN,
    max_prot_len=MAX_PROT_LEN
    )

eval_dtst = edt.Eval_Protein_dataset(
    root_dir="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/raw/model1/test",
    min_prot_len=MIN_PROT_LEN,
    max_prot_len=MAX_PROT_LEN)


# train_inhibit = cdt.Inhibitor_dataset(
#     root_dir="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/raw/model2",
#     min_prot_len=MIN_PROT_LEN,
#     max_prot_len=MAX_PROT_LEN
# )

# test_inhibit = edt.Eval_Inhibitor_dataset(
#     root_dir="/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/raw/model2",
#     min_prot_len=MIN_PROT_LEN,
#     max_prot_len=MAX_PROT_LEN
# )

torch.save(train_dtst,"/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/save/model1/train.pt")
torch.save(eval_dtst,"/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/save/model1/test.pt")
# torch.save(train_inhibit,"/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/save/model2/train.pt")
# torch.save(test_inhibit, "/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/data/save/model2/test.pt")