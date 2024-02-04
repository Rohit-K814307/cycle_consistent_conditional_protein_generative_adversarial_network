from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
from io import StringIO
import torch

def rmsd(pdb1, pdb2):
    # Parse the PDB strings
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('1', StringIO(pdb1))
    structure2 = parser.get_structure('2', StringIO(pdb2))

    # Extract the coordinates of the atoms
    coords1 = []
    coords2 = []

    for model in structure1:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords1.append(atom.get_coord())

    for model in structure2:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords2.append(atom.get_coord())

    # Create a SVDSuperimposer object
    superimposer = SVDSuperimposer()

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    if len(coords1) < len(coords2):
        coords1 = coords1[0:len(coords1)]
        coords2 = coords2[0:len(coords1)]
    else:
        coords1 = coords1[0:len(coords2)]
        coords2 = coords2[0:len(coords2)]


    # Set the coordinates to be superimposed
    superimposer.set(coords1, coords2)

    # Perform the superimposition
    superimposer.run()

    # Get the RMSD value
    rmsd = superimposer.get_rms()

    return rmsd



def avg_rmsd(pdbs1, pdbs2):
    summed_rmsd = 0

    for i in range(len(pdbs1)):
        pdba = pdbs1[i]
        pdbb = pdbs2[i]

        summed_rmsd += rmsd(pdba, pdbb)

    return summed_rmsd / len(pdbs1)

    


def seq_feasibility(preds, actuals):
    """

    Parameters:

        preds: predictions (onehot)

        actuals: actual y-hats

    """

    return (((preds == actuals).sum().item()) / (preds == actuals).size(0))
    
def hamming_distance(chain1, chain2):
    return sum(c1 != c2 for c1, c2 in zip(chain1, chain2))


def seq_diversity(preds):
    """Finds average hamming distance of sequences in dataset
    
    Parameters:
    
        preds -- predictions in onehot encoded format
        
    """

    preds = torch.argmax(preds, dim=-1).numpy()

    hamming_sum = 0
    num_comparisons = 0

    for j in range(len(preds)):
        for k in range(j, len(preds)):
            hamming_sum += hamming_distance(preds[j],preds[k])
            num_comparisons += 1
    
    return hamming_sum/num_comparisons



    