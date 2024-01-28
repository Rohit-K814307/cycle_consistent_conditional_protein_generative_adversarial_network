import torch
from io import StringIO
from Bio.PDB import PDBParser
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer


def rmsd(pdb1, pdb2):
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("prot", StringIO(pdb1))
    structure2 = parser.get_structure("prot", StringIO(pdb2))


    atoms1 = []
    atoms2 = []
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for residue1, residue2 in zip(chain1, chain2):
                for atom1, atom2 in zip(residue1, residue2):
                    if atom1.get_id() == atom2.get_id():
                        atoms1.append(atom1.coord)
                        atoms2.append(atom2.coord)


    superimposer = SVDSuperimposer()

    atoms1 = np.array(atoms1)
    atoms2 = np.array(atoms2)

    superimposer.set(atoms1, atoms2)
    superimposer.run()

    rmsd = superimposer.get_rms()

    return rmsd


def pairwise_avg_rmsd(pdbs):

    summed_rmsd = 0
    pairwise_comps = 0

    for j in range(len(pdbs)):
        for k in range(j+1, len(pdbs)):

            summed_rmsd += rmsd(pdbs[j], pdbs[k])
            pairwise_comps += 1

    return summed_rmsd / pairwise_comps


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



    