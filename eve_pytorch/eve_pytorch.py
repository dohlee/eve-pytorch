import torch
import torch.nn as nn

class EVE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

if __name__ == '__main__':
    model = EVE()
    