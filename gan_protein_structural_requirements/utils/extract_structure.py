from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

def extract_c8_dssp(pdb_file_path):
    # Create a PDB parser
    parser = PDBParser(QUIET=True)

    # Load the PDB file and get the structure
    structure = parser.get_structure('protein', pdb_file_path)

    # Initialize a dictionary to store C8 DSSP assignments
    c8_dssp = {}

    for model in structure:
        for chain in model:
            dssp = DSSP(chain, pdb_file_path, dssp='mkdssp')

            for key, ss in dssp.property_dict.items():
                c8_ss = 'C'  # Default is coil (C)

                if ss == 'H':
                    c8_ss = 'H'  # Helix
                elif ss == 'E':
                    c8_ss = 'E'  # Strand

                c8_dssp[key] = c8_ss

    return c8_dssp

# Example usage:
pdb_file_path = '../data/batch_1_data/1et1.pdb'
c8_dssp = extract_c8_dssp(pdb_file_path)
print("C8 DSSP Secondary Structure:", c8_dssp)
