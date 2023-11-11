import os

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from utils.polarity_list import polarity_list

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
    
# extract_structures("../data/raw/batch_1_data")