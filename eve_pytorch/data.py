import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from Bio import SeqIO

a2i = {a:i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

def one_hot_encode_amino_acid(sequence):
    return torch.eye(len(a2i))[[a2i[a] for a in sequence]].T

class MSADataset(Dataset):
    def __init__(self, records):
        """MSA Dataset
        records: List of SeqRecord objects.
        """

        self.data, self.weights = [], []
        for record in records:
            self.data.append(one_hot_encode_amino_acid(record.seq))
            self.weights.append(float(record.id.split('@@@')[1]))

        self.seq_len = self.data[0].shape[1]
    
    def _hamming(self, i, j):
        return 1.0 - torch.sum(self.data[i] * self.data[j]) / self.seq_len
    
    def get_Neff(self):
        """Return the effective number of sequences in the MSA.
        """
        return sum(self.weights)
    
    def get_sampling_weights(self):
        return self.weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = MSADataset('../data/P53_HUMAN_b01.filtered.a2m')

    print('Neff', ds.get_Neff())

    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        print(batch.shape)
        break