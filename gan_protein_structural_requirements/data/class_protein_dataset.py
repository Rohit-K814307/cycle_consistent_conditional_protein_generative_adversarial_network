import torch
from torch.utils.data import Dataset, DataLoader
from utils import protein_visualizer as viz
from utils import extract_structure as struct
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

class Protein_dataset(Dataset):

    def __init__(self, root_dir, transforms=None):
        """
        Arguments:
            root_dir (string): Directory containing PDB files
        """
        self.root_dir = root_dir
        features, labels = self.aggregate_data()
        self.X = features
        self.Y = labels

        print(self.X, self.Y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def process_polarity(self,pol_str):
        return (pol_str.count('1') + pol_str.count('2')) / len(pol_str)
    
    def process_c8(self, c8_str):
        return c8_str

    def aggregate_data(self):
        structures = struct.extract_structures(self.root_dir)
        features = []
        labels = []
        for id in structures["secondary"].keys():
            content_sec = structures["secondary"][id]
            content_pol = structures["primary_pol"][id]

            feature = [self.process_c8(content_sec['c8']), self.process_polarity(content_pol['polarity_conv'])]
            features.append((id, feature))

            label = content_sec["sequence"]
            labels.append(label)

        return features, labels

    def vizualize_p(self, entry, save_path=None):
        for ex in self.X:
            if entry == ex[1]:
                id = ex[0]
                viz.viz_file(id=id, 
                             fdir=self.root_dir, 
                             show=False,
                             save_path=save_path)


p = Protein_dataset("data/raw/batch_1_data")

