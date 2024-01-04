import torch
from torch.utils.data import Dataset
from ..utils import extract_structure as struct
from ..utils import folding_models as fold

class Protein_dataset(Dataset):

    def __init__(self, root_dir,  min_prot_len, max_prot_len):
        """
        Parameters

            root_dir (string): Directory containing PDB files

            min_prot_len (int): minimum length of proteins to filter through

            max_prot_len (int): maximum length of proteins to filter through

        """

        #add protein ids for tracking
        self.ids = []

        self.root_dir = root_dir
        self.min_prot_len = min_prot_len
        self.max_prot_len = max_prot_len
        features, labels = self.aggregate_data()
        inps, outs = self.upsample(features, labels)

        #shape = (batch_size, number of design objectives)
        self.inps = inps
        self.X = torch.FloatTensor(inps).unsqueeze(1).repeat(1,self.max_prot_len,1)

        #shape = (batch_size, sequence max length, number of amino acids)
        self.Y, self.encode_cats, self.decode_cats = self.onehot_encode(outs)
        

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        ids = self.ids[idx]

        return {"X":x,"Y":y,"IDS":ids}

    def pad_label(self, sequence, maxlen):
        for _ in range(maxlen - len(sequence)):
            t = [0] * len(sequence[0])
            t[-1] = 1
            sequence.append(t)

        return sequence

    def onehot_encode(self, labels):
        categories = fold.get_vocab_encodings()
        cat_dict = {categories[i] : i for i in range(len(categories)) if categories[i] != "X"}
        cat_dict["<pad>"] = 20
        decode_dict = {cat_dict[key] : key for key in cat_dict.keys()}
        encoded_data = []

        for example in labels:
            datapoint = []

            for value in example:
                encoded_val = [0] * int(len(cat_dict.keys()))
                if value != "X":
                    encoded_val[cat_dict.get(value)] = 1
                else:
                    pass
                datapoint.append(encoded_val)

            datapoint = self.pad_label(datapoint, self.max_prot_len)
            encoded_data.append(datapoint)
        return torch.tensor(encoded_data), cat_dict, decode_dict

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
                labels.append(list(content_sec["sequence"]))
                self.ids.append(id)

        return features, labels
    
    def upsample(self, X, Y):

        y = []
        x = []
        #upsample from classes that are not as even
        sorted_list = sorted(X, key=lambda x: max(x[1:8]), reverse=True)
        x += sorted_list[:50]

        for entry in x:
            y.append(Y[X.index(entry)])
            self.ids.append(self.ids[X.index(entry)])

        X.extend(x)
        Y.extend(y)

        return X, Y