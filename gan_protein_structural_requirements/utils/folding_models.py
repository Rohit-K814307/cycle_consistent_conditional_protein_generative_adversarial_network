import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import os
import numpy as np

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path)

def load_esm(path,verbose=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path)
    model = EsmForProteinFolding.from_pretrained(path, low_cpu_mem_usage=True)

    if verbose:
        print(model)

    return model, tokenizer

def esm_predict(model, tokenized_input):
    with torch.no_grad():
        output = model(tokenized_input)

    pdb = convert_outputs_to_pdb(output)

    return output, pdb

def esm_batch_predict(seqs, model):
    preds = []
    for seq in seqs:
        with torch.no_grad():
            output = model(seq)

        pdb = convert_outputs_to_pdb(output)
        preds.append((output, pdb))

    return preds

def get_vocab_encodings():
    path = os.getcwd() + "/gan_protein_structural_requirements/utils/esmfoldv1/vocab.txt"
    arr = []

    with open(path,'r') as file:
        for _ in range(21):
            line = file.readline().strip()
            arr.append(line[0])
    
    return arr