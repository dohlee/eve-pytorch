import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
import os
import wandb

from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from Bio import SeqIO

from eve_pytorch import EVE
from eve_pytorch.data import MSADataset

ALPHABET_SIZE = 21

def get_sequence_length(a2m_fp):
    """Get the sequence length of the first sequence in the a2m file.
    a2m_fp: Path to a2m file.
    """
    for record in SeqIO.parse(a2m_fp, "fasta"):
        return len(record.seq)

def cycle(loader, n):
    """Cycle through a dataloader indefinitely."""
    cnt, stop_flag = 0, False
    while stop_flag is False:
        for batch in loader:
            yield batch

            cnt += 1
            if cnt == n:
                stop_flag = True
                break

def train(model, train_loader, optimizer, num_steps):
    model.train()

    # Training loop with progressbar.
    bar = tqdm.tqdm(cycle(train_loader, n=num_steps), total=num_steps, leave=False)
    for idx, seq in enumerate(bar):
        seq = seq.cuda()
        bsz = seq.shape[0]

        optimizer.zero_grad()
        seq_recon, z_mu, z_log_var = model(seq, return_latent=True)

        ce_loss = F.cross_entropy(seq_recon.view(-1, ALPHABET_SIZE), seq.argmax(dim=1).flatten(), reduction='sum') / bsz
        z_kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp()) / bsz
        w_kl_loss = model.get_w_kl() / 2000  # Assume Neff=2000 For now

        warmup_scale = 1.0
        z_kl_scale = 1.0
        w_kl_scale = 1.0

        loss = ce_loss + warmup_scale * (z_kl_scale * z_kl_loss + w_kl_scale * w_kl_loss) # TODO: implement loss scales
        loss.backward()
        optimizer.step()

        bar.set_postfix(
            loss=f'{loss.item():.4f}',
            ce=f'{ce_loss.item():.4f}',
            z_kl=f'{z_kl_loss.item():.4f}',
            w_kl=f'{w_kl_loss.item():.4f}'
        )

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Performance drops, so commenting out for now.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def main():
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--msa', help='Path to MSA file (in a2m)', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--num-steps', '-n', type=int, default=400_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)  # Supp. Note 3.2.2
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    if not args.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    
    wandb.init(project='eve-pytorch', config=args, reinit=True)
    train_set = MSADataset(a2m_fp=args.msa)

    # TODO: sample sequences according to sampling weights = 1 / (# seqs with hamming < th)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    model = EVE(seq_len=get_sequence_length(args.msa), alphabet_size=ALPHABET_SIZE)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    train(model, train_loader, optimizer, num_steps=args.num_steps)

if __name__ == '__main__':
    main()
