import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
import os
import wandb

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

def validate(model, val_loader, val_Neff):
    model.eval()

    # Validation loop.
    losses = []
    ce_losses = []
    z_kl_losses = []
    w_kl_losses = []

    with torch.no_grad():
        for idx, seq in enumerate(val_loader):
            seq = seq.cuda()
            bsz = seq.shape[0]

            seq_recon, z_mu, z_log_var = model(seq, return_latent=True)

            ce_loss = F.cross_entropy(seq_recon.view(-1, ALPHABET_SIZE), seq.argmax(dim=1).flatten(), reduction='sum') / bsz
            z_kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp()) / bsz
            w_kl_loss = model.get_w_kl() / val_Neff

            # loss := (negative ELBO) = - [(log-likelihood - z_kl) - w_kl / Neff] = (negative log-likelihood) + z_kl + w_kl / Neff
            loss = ce_loss + z_kl_loss + w_kl_loss

            losses.append(loss.item())
            ce_losses.append(ce_loss.item())
            z_kl_losses.append(z_kl_loss.item())
            w_kl_losses.append(w_kl_loss.item())

    model.train()
    return np.mean(losses), np.mean(ce_losses), np.mean(z_kl_losses), np.mean(w_kl_losses)

def train(model, train_loader, val_loader, optimizer, num_steps, train_Neff, val_Neff, best_model_fp):
    model.train()

    best_val_loss = np.inf

    # Training loop with progressbar.
    bar = tqdm.tqdm(cycle(train_loader, n=num_steps), total=num_steps, leave=False)
    for idx, seq in enumerate(bar):
        seq = seq.cuda()
        bsz = seq.shape[0]

        optimizer.zero_grad()
        seq_recon, z_mu, z_log_var = model(seq, return_latent=True)

        ce_loss = F.cross_entropy(seq_recon.view(-1, ALPHABET_SIZE), seq.argmax(dim=1).flatten(), reduction='sum') / bsz
        z_kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp()) / bsz
        w_kl_loss = model.get_w_kl() / train_Neff

        warmup_scale = 1.0
        z_kl_scale = 1.0
        w_kl_scale = 1.0

        # loss := (negative ELBO) = - [(log-likelihood - z_kl) - w_kl / Neff] = (negative log-likelihood) + z_kl + w_kl / Neff
        loss = ce_loss + warmup_scale * (z_kl_scale * z_kl_loss + w_kl_scale * w_kl_loss) # TODO: implement loss scales
        loss.backward()
        optimizer.step()

        if idx % 1000 == 0:
            bar.set_postfix(
                loss=f'{loss.item():.4f}',
                ce=f'{ce_loss.item():.4f}',
                z_kl=f'{z_kl_loss.item():.4f}',
                w_kl=f'{w_kl_loss.item():.4f}'
            )

            wandb.log({
                'train/loss': loss.item(),
                'train/ce': ce_loss.item(),
                'train/z_kl': z_kl_loss.item(),
                'train/w_kl': w_kl_loss.item(),
            })

            val_loss, val_ce_loss, val_z_kl_loss, val_w_kl_loss = validate(model, val_loader, val_Neff)
            wandb.log({
                'val/loss': val_loss,
                'val/ce': val_ce_loss,
                'val/z_kl': val_z_kl_loss,
                'val/w_kl': val_w_kl_loss
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_fp)

                print('Saved best model.')

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

    records = []
    for record in SeqIO.parse(args.msa, "fasta"):
        records.append(record)

    random.shuffle(records)
    train_records = records[:int(0.8 * len(records))]
    val_records = records[int(0.8 * len(records)):]  # 80:20 train-val split.

    train_set = MSADataset(records=train_records)
    val_set = MSADataset(records=val_records)

    train_Neff = train_set.get_Neff()
    val_Neff = val_set.get_Neff()

    sampler = WeightedRandomSampler(
        weights=train_set.get_sampling_weights(),
        num_samples=len(train_set),
        replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=16,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=16,
        pin_memory=True
    )

    model = EVE(seq_len=get_sequence_length(args.msa), alphabet_size=ALPHABET_SIZE)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    train(model, train_loader, val_loader, optimizer, num_steps=args.num_steps, train_Neff=train_Neff, val_Neff=val_Neff, best_model_fp=args.output)

if __name__ == '__main__':
    main()
