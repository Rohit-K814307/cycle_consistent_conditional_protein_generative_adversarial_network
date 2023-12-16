import torch
from torch.utils.data import Dataset, DataLoader
from ..utils import protein_visualizer as viz
from ..utils import extract_structure as struct
from ..utils import folding_models as fold
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

class Protein_dataset(Dataset):

    def __init__(self, root_dir, path_to_esm, min_prot_len, max_prot_len, transforms=None):
        """
        Arguments:
            root_dir (string): Directory containing PDB files
        """
        self.root_dir = root_dir
        self.min_prot_len = min_prot_len
        self.max_prot_len = max_prot_len
        self.path_to_esm = path_to_esm
        features, labels = self.aggregate_data()
        inps, outs = self.upsample(features, labels)
        self.X = torch.FloatTensor(inps)
        self.Y = torch.stack(outs)
        

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def pad_label(self, sequence, maxlen):
        for _ in range(maxlen - len(sequence)):
            sequence += "<pad>"

        return sequence




    def aggregate_data(self):
        structures = struct.extract_structures(self.root_dir)
        features = []
        labels = []
        for id in structures["secondary"].keys():
            
            content_sec = structures["secondary"][id]
            content_pol = structures["primary_pol"][id]

            if len(content_sec["sequence"]) > self.min_prot_len and len(content_sec["sequence"]) < self.max_prot_len:
                feature = struct.convert_dssp_string(content_sec['c8']) + [struct.convert_pol_string(content_pol['polarity_conv'])]
                features.append(feature)

                label = self.pad_label(content_sec["sequence"], self.max_prot_len)
                tokenizer = fold.load_tokenizer(self.path_to_esm)
                tokenized_label = tokenizer([label], return_tensors="pt", add_special_tokens=False)['input_ids']
                labels.append(tokenized_label[0])

        return features, labels
    
    def upsample(self, X, Y):

        y = []
        x = []
        #upsample from classes that are not as even
        sorted_list = sorted(X, key=lambda x: max(x[1:8]), reverse=True)
        x += sorted_list[:50]

        for entry in x:
            y.append(Y[X.index(entry)])

        X.extend(x)
        Y.extend(y)

        return X, Y
        

