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

def get_sequence_length(a2m_fp):
    """Get the sequence length of the first sequence in the a2m file.
    a2m_fp: Path to a2m file.
    """
    for record in SeqIO.parse(a2m_fp, "fasta"):
        return len(record.seq)

def train(model, train_loader, optimizer, criterion, metrics_f):
    model.train()
    running_output, running_label = [], []

    # Training loop with progressbar.
    bar = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
    for idx, batch in enumerate(bar):
        seq = batch['seq'].cuda()

        optimizer.zero_grad()
        output, z_mu, z_log_var = model(seq, return_latent=True).flatten()

        ce_loss = F.cross_entropy(seq, seq.argmax(dim=1))
        z_kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        z_weight_loss = model.get_kl_w() / 2000  # For now

        loss = ce_loss + z_kl_loss + z_weight_loss
        loss.backward()
        optimizer.step()

        running_output.append(output.detach().cpu())
        running_label.append(label.detach().cpu())

        if idx % 100 == 0:
            running_output = torch.cat(running_output, dim=0)
            running_label = torch.cat(running_label, dim=0)

            running_loss = criterion(running_output, running_label)
            running_metrics = {k: f(running_output, running_label) for k, f in metrics_f.items()}

            loss = running_loss.item()
            pearson = running_metrics['pearson']
            spearman = running_metrics['spearman']
            bar.set_postfix(loss=loss, pearson=pearson, spearman=spearman)
            wandb.log({
                'train/loss': loss,
                'train/pearson': pearson,
                'train/spearman': spearman,
            })

            running_output, running_label = [], []

def validate(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'val/loss': loss,
        'val/pearson': metrics['pearson'],
        'val/spearman': metrics['spearman'],
        'val/pearson_fr': metrics['pearson_fr'],
        'val/delta': metrics['delta'],
    })

    return loss, metrics

def test(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'test/loss': loss,
        'test/pearson': metrics['pearson'],
        'test/spearman': metrics['spearman'],
        'test/pearson_fr': metrics['pearson_fr'],
        'test/delta': metrics['delta'],
    })

    return loss, metrics

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
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=72)  
    parser.add_argument('--lr', type=float, default=1e-4)  # Supp. Note 3.2.2
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    if not args.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    
    wandb.init(project='eve-pytorch', config=args, reinit=True)

    train_df = pd.read_csv(args.train)
    train_set = EVE(alphabet_size=21, seq_len=get_sequence_length(args.msa))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    model = EVE()
    model = model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, metrics_f)
        scheduler.step()
    
    wandb.log({
        'best_val_loss': best_val_loss,
        'best_val_pearson': best_val_pearson,
        'best_val_spearman': best_val_spearman,
        'test_loss': best_test_loss,
        'test_pearson': best_test_pearson,
        'test_spearman': best_test_spearman,
    })

if __name__ == '__main__':
    main()
