import os

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from .polarity_list import polarity_list

import tempfile

def extract_secondary_structure(fnames, fpaths):
    """
    Arguments:
    
        fnames (list): Names of files in the dir
        fpaths (list): Paths of files corresponding to fnames
    """

    structures = {}
    bad_ids = []
    #secondary structure (c8 OR c3 dssp) - we can use either doesnt really matter just collect both for now
    for idx in range(len(fnames)):
        fname = fnames[idx]
        fpath = fpaths[idx]
        try:
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
        except:
            bad_ids.append(fname)
    return structures, bad_ids


def find_seq_polarity(sequence):
    new_seq = ""
    po = polarity_list()

    for s in sequence:
        if s in po.keys():
            new_seq += str(po.get(s).get("polarity_type"))
        else:
            new_seq += "0"

    return new_seq


def extract_primary_polarity(fnames, fpaths):
    """
    Arguments:
    
        fnames (list): Names of files in the dir
        fpaths (list): Paths of files corresponding to fnames
    """

    polarities = {}
    bad_ids = []

    for idx in range(len(fnames)):
        fname = fnames[idx]
        fpath = fpaths[idx]

        try:
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
        except:
            bad_ids.append(fname)
    return polarities, bad_ids

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
            tmp.write("".join(pdb_batch[i][1]).encode('utf-8'))
            structure = p.get_structure(f"Protein_{i}",tmp.name)
            model = structure[0]
            dssp = DSSP(model, tmp.name)
            out.append(dssp)
        finally:
            tmp.close()  # deletes the file
    return out

def polarity_content_batch(batch_seq):
    """
    Arguments:
    
        batch_seq (list): list of sequences/primary structure
    """

    polarities = []
    for sequence in batch_seq:
        polarity_conv = find_seq_polarity(sequence)
        polarities.append(polarity_conv)

    return polarities

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

    secondary_structure, _ = extract_secondary_structure(fnames, fpaths)
    structures["secondary"] = secondary_structure
    
    polarity, _ = extract_primary_polarity(fnames, fpaths)
    structures["primary_pol"] = polarity

    return structures
    
def vocab_list():
    vocab = []
    for nam in polarity_list().keys():
        vocab.append(nam)
    return vocab

def get_untokenizer(path):
    path_to_vocab = path

    with open(path_to_vocab, 'r') as file:
        lines = file.readlines()
    
    result_dict = {i: value.strip() for i, value in enumerate(lines)}
    result_dict = {key: value for key, value in result_dict.items()}

    return result_dict

def untokenize(untokenizer, batch):
    result = []
    for token_seq in batch:
        result_i = ""
        for token_num in token_seq:
            result_i += untokenizer.get(token_num.item())
        result.append(result_i)
    return result

def convert_dssp_string(dssp):
    """
    Arguments:
        dssp (string): string of dssp content (c8)
    
    Returns:
        dssp percentage composition (list): respective indices and content characteristics:
            0 - alpha helix
            1 - isolated beta-bridge residue
            2 - strand
            3 - 3-10 helix
            4 - pi helix
            5 - turn
            6 - bend
            7 - none
    """
    values = ["H","B","E","G","I","T","S","~"]
    content = []

    for val in values:
        content.append(dssp.count(val)/len(dssp))

    return content

def convert_pol_string(pol):
    percent_polarity = 0

    for val in pol:
        if int(val) > 0:
            percent_polarity += 1

    percent_polarity /= len(pol)

    return percent_polarity



#############test examples########################
# from utils.folding_models import esm_batch_predict, load_esm
# model, tokenizer = load_esm(verbose=False)
# print("model loaded \n\n\n\n\n\n\n\n")
# test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"
# batch = [test_protein]
# outs = esm_batch_predict(batch, model, tokenizer)
# print("prediction complete \n\n\n\n\n\n\n\n\n")
# out_dssp = dssp_pdb(outs)
# dssp_content_batch = dssp_content_object_batch(out_dssp)
# print(dssp_content_batch)
# """
# outputs:
# [('~~~~~~HHHHHHHHHHHHHHHHHHHHHTEEEEEEE~STTSSHHHHHHHHHHHHS~S~~HHHHHHHHHHHHHHHHHHHHHHHHHHHHTT~~~SSTTHHHHHHHHHHHHTT~~TT~~~HHHHHHHHHHHTSHHHHHHHTTGGGS~~~TTHHHHHHTHHHHTSTT~~~~HHHHHH~~~~~~SEEEEEEEETTEEEEEEEE~~STTTGGGGGGG~TT~SEEEEEEEGGGTT~B~SS~TTSBHHHHHHHHHHHHHT~GGGSSSEEEEEEE~HHHHHHHTTTS~GGGT~TT~~S~SSHHHHHHHHHHHHHHT~TTTTT~~EEEEE~~TT~HHHHHHHHHHHHHHHHHHHHHHTT~~', '~~~~~~HHHHHHHHHHHHHHHHHHHHH~EEEEEEE~~~~~~HHHHHHHHHHHH~~~~~HHHHHHHHHHHHHHHHHHHHHHHHHHHH~~~~~~~~~HHHHHHHHHHHH~~~~~~~~~HHHHHHHHHHH~~HHHHHHH~~HHH~~~~~~HHHHHH~HHHH~~~~~~~~HHHHHH~~~~~~~EEEEEEEE~~EEEEEEEE~~~~~~HHHHHHH~~~~~EEEEEEEHHH~~~E~~~~~~~EHHHHHHHHHHHHH~~HHH~~~EEEEEEE~HHHHHHH~~~~~HHH~~~~~~~~~~HHHHHHHHHHHHHH~~~~~~~~~EEEEE~~~~~HHHHHHHHHHHHHHHHHHHHHH~~~~')]
# """