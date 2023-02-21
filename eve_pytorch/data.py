import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from Bio import SeqIO

a2i = {a:i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

def one_hot_encode_amino_acid(sequence):
    return torch.eye(len(a2i))[[a2i[a] for a in sequence]].T

class MSADataset(Dataset):
    def __init__(self, a2m_fp):
        """MSA Dataset
        a2m_fp: Path to a2m file.
        """

        self.data = []
        for record in SeqIO.parse(a2m_fp, "fasta"):
            self.data.append(one_hot_encode_amino_acid(record.seq))

        self.seq_len = self.data[0].shape[1]
        # self._compute_sampling_weights()
        # self._compute_neff()
    
    def _hamming(self, i, j):
        return 1.0 - torch.sum(self.data[i] * self.data[j]) / self.seq_len

    def _compute_sampling_weights(self):
        """Compute sampling weights for each sequence in the MSA.
        For ith sequence, w_i = 1 / (# sequences in MSA that is hamming_dist < 0.2)
        """
        self.weights = []
        for i in range(len(self.data)):
            cnt = 0
            for j in range(len(self.data)):
                if i == j:
                    continue
                
                if self._hamming(i, j) < 0.2:
                    cnt += 1
            
            self.weights.append(1.0 / cnt)
    
    def _compute_Neff(self):
        """Compute effective number of sequences in the MSA.
        """
        self.Neff = sum(self.weights)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = MSADataset('../data/P53_HUMAN_b01.filtered.a2m')

    print(ds.Neff)

    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        print(batch.shape)
        break