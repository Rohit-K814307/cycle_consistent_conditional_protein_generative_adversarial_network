import torch
from torch.utils.data import Dataset, DataLoader
from ..utils import protein_visualizer as viz
from ..utils import extract_structure as struct
from ..utils import folding_models as fold
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

class Protein_dataset(Dataset):

    def __init__(self, root_dir, path_to_esm, transforms=None):
        """
        Arguments:
            root_dir (string): Directory containing PDB files
        """
        self.root_dir = root_dir
        #"/Users/rohitkulkarni/Documents/gan_protein_structural_requirements/gan_protein_structural_requirements/utils/esmfoldv1"
        self.path_to_esm = path_to_esm
        features, labels = self.aggregate_data()
        self.X = features
        self.Y = labels

        print(self.X, self.Y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def aggregate_data(self):
        structures = struct.extract_structures(self.root_dir)
        features = []
        labels = []
        for id in structures["secondary"].keys():
            content_sec = structures["secondary"][id]
            content_pol = structures["primary_pol"][id]

            feature = struct.convert_dssp_string(content_sec['c8']) + [struct.convert_pol_string(content_pol['polarity_conv'])]
            features.append(feature)

            label = content_sec["sequence"]
            tokenizer = fold.load_tokenizer(self.path_to_esm)
            tokenized_label = tokenizer([label], return_tensors="pt", add_special_tokens=False)['input_ids']
            labels.append(tokenized_label[0])

        return features, labels

