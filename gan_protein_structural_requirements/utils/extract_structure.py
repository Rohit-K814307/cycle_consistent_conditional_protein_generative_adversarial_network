import os

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from utils.polarity_list import polarity_list

import tempfile

def extract_secondary_structure(fnames, fpaths):
    """
    Arguments:
    
        fnames (list): Names of files in the dir
        fpaths (list): Paths of files corresponding to fnames
    """

    structures = {}

    #secondary structure (c8 OR c3 dssp) - we can use either doesnt really matter just collect both for now
    for idx in range(len(fnames)):
        fname = fnames[idx]
        fpath = fpaths[idx]

        p = PDBParser(QUIET=True)
        structure = p.get_structure(fname, fpath)
        model = structure[0]
        dssp = DSSP(model, fpath)

        # extract sequence and secondary structure from the DSSP tuple
        sequence = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
            sec_structure += dssp[a_key][2]

        #
        # The DSSP codes for secondary structure used here are:
        # =====     ====
        # Code      Structure
        # =====     ====
        # H         Alpha helix (4-12)
        # B         Isolated beta-bridge residue
        # E         Strand
        # G         3-10 helix
        # I         Pi helix
        # T         Turn
        # S         Bend
        # ~         None
        # =====     ====
        #
        
        sec_structure = sec_structure.replace('-', '~')
        sec_structure_3state=sec_structure

        # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
        sec_structure_3state = sec_structure_3state.replace('H', 'H') #0
        sec_structure_3state = sec_structure_3state.replace('E', 'E')
        sec_structure_3state = sec_structure_3state.replace('T', '~')
        sec_structure_3state = sec_structure_3state.replace('~', '~')
        sec_structure_3state = sec_structure_3state.replace('B', 'E')
        sec_structure_3state = sec_structure_3state.replace('G', 'H') #5
        sec_structure_3state = sec_structure_3state.replace('I', 'H') #6
        sec_structure_3state = sec_structure_3state.replace('S', '~')


        structures[fname] = {"sequence":sequence, "c8":sec_structure, "c3":sec_structure_3state}

    return structures


def find_seq_polarity(sequence):
    new_seq = ""
    po = polarity_list()

    for s in sequence:
        if s in po.keys():
            new_seq += str(po.get(s).get("polarity_type"))
        else:
            new_seq += "~"

    return new_seq


def extract_primary_polarity(fnames, fpaths):
    """
    Arguments:
    
        fnames (list): Names of files in the dir
        fpaths (list): Paths of files corresponding to fnames
    """

    polarities = {}

    for idx in range(len(fnames)):
        fname = fnames[idx]
        fpath = fpaths[idx]

        p = PDBParser(QUIET=True)
        structure = p.get_structure(fname, fpath)
        model = structure[0]
        dssp = DSSP(model, fpath)

        # extract sequence from DSSP tuple
        sequence = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
        
        #find polarity seq from AA seq
        polarity_conv = find_seq_polarity(sequence)

        polarities[fname] = {"sequence":sequence, "polarity_conv":polarity_conv}

    return polarities

def dssp_pdb(pdb_batch):
    """
    Arguments:
    
        pdb_batch (list): batch of pdb object outputs from folding_models.esm_batch_predict()
    """
    out = []
    p = PDBParser(QUIET=True)

    for i in range(len(pdb_batch)):
        tmp = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb")
        try:

            #TODO:fix the encoding so it opens the file 
            # and writes to it with proper encoding rather
            #  than doing this (if possible)

            tmp.write("".join(pdb_batch[i][1]).encode('utf-8'))
            structure = p.get_structure(f"Protein_{i}",tmp.name)
            model = structure[0]
            dssp = DSSP(model, tmp.name)
            out.append(dssp)
        finally:
            tmp.close()  # deletes the file
    return out

def dssp_content_object_batch(batch_dssp):
    """
    Arguments:
    
        batch_dssp (list): batch of DSSP objects
    """

    dssp_content_batch = []

    for dssp in batch_dssp:
        sequence = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
            sec_structure += dssp[a_key][2]

        #
        # The DSSP codes for secondary structure used here are:
        # =====     ====
        # Code      Structure
        # =====     ====
        # H         Alpha helix (4-12)
        # B         Isolated beta-bridge residue
        # E         Strand
        # G         3-10 helix
        # I         Pi helix
        # T         Turn
        # S         Bend
        # ~         None
        # =====     ====
        #
        
        sec_structure = sec_structure.replace('-', '~')
        sec_structure_3state=sec_structure

        # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
        sec_structure_3state = sec_structure_3state.replace('H', 'H') #0
        sec_structure_3state = sec_structure_3state.replace('E', 'E')
        sec_structure_3state = sec_structure_3state.replace('T', '~')
        sec_structure_3state = sec_structure_3state.replace('~', '~')
        sec_structure_3state = sec_structure_3state.replace('B', 'E')
        sec_structure_3state = sec_structure_3state.replace('G', 'H') #5
        sec_structure_3state = sec_structure_3state.replace('I', 'H') #6
        sec_structure_3state = sec_structure_3state.replace('S', '~')

        dssp_content_batch.append((sec_structure, sec_structure_3state))
    
    return dssp_content_batch


def extract_structures(dir_path):
    """
    Arguments:
    
        dir_path (string): Root path of PDB files
    """

    structures = {}

    #get file name and full path
    fnames = []
    fpaths = []
    for file in os.listdir(dir_path):
        if ".pdb" in file:
            fnames.append(file.split(".")[0])
            fpaths.append(os.path.join(dir_path, file))

    secondary_structure = extract_secondary_structure(fnames, fpaths)
    structures["secondary"] = secondary_structure
    
    polarity = extract_primary_polarity(fnames, fpaths)
    structures["primary_pol"] = polarity


    return structures
    
# from utils.folding_models import esm_batch_predict, load_esm
# model, tokenizer = load_esm(verbose=False)
# print("model loaded \n\n\n\n\n\n\n\n")
# batch = ["GVGVGVGVGVGVGVGVGVVVG", "VVVGVGVGGGGVG", "GVVVVGVGVGVGV"]
# outs = esm_batch_predict(batch, model, tokenizer)
# print("prediction complete \n\n\n\n\n\n\n\n\n")
# out_dssp = dssp_pdb(outs)
# dssp_content_batch = dssp_content_object_batch(out_dssp)