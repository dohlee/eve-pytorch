import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange

#     "training_parameters": {
#         "num_training_steps"                :   400000,
#         "learning_rate"                     :   1e-4,
#         "batch_size"                        :   256,
#         "annealing_warm_up"                 :   0,
#         "kl_latent_scale"                   :   1.0,
#         "kl_global_params_scale"            :   1.0,
#         "l2_regularization"                 :   0.0,
#         "use_lr_scheduler"                  :   false,
#         "use_validation_set"                :   false,
#         "validation_set_pct"                :   0.2,
#         "validation_freq"                   :   1000,
#         "log_training_info"                 :   true,
#         "log_training_freq"                 :   1000,
#         "save_model_params_freq"            :   500000
#     }
# }

def sample(mean, log_var):
    mu = torch.zeros_like(mean)
    sigma = torch.ones_like(log_var)
    eps = torch.normal(mu, sigma).cuda()
    return torch.exp(0.5 * log_var) * eps + mean

class Encoder(nn.Module):
    def __init__(self, alphabet_size, seq_len):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(alphabet_size * seq_len, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 300),
            nn.ReLU(),
        )

        self.to_z_mu = nn.Linear(300, 50)
        self.to_z_log_var = nn.Linear(300, 50)

        nn.init.constant_(self.to_z_mu.bias, 0.1)
        nn.init.constant_(self.to_z_log_var.bias, -10.0)

    def forward(self, x):
        x = rearrange(x, 'b c l -> b (c l)')

        x = self.stem(x)
        z_mu = self.to_z_mu(x)
        z_log_var = self.to_z_mu(x)

        return z_mu, z_log_var

class DecoderLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, xavier_init=False):
        super().__init__()
        self.bias = bias

        self.w_mu = nn.Linear(in_features, out_features, bias=bias)
        if xavier_init:
            nn.init.xavier_normal_(self.w_mu.weight)
        elif self.bias:
            nn.init.constant_(self.w_mu.bias, 0.1)

        self.w_log_var = nn.Linear(in_features, out_features, bias=bias)
        nn.init.constant_(self.w_log_var.weight, -10.0)
        if self.bias:
            nn.init.constant_(self.w_log_var.bias, -10.0)

    def forward(self, z):
        w = sample(self.w_mu.weight, self.w_log_var.weight)

        if self.bias:
            b = sample(self.w_mu.bias, self.w_log_var.bias)
            return F.linear(z, weight=w, bias=b)
        else:
            return F.linear(z, weight=w)
        
class AddBias(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(in_features) * 0.1)
        self.log_var = nn.Parameter(torch.ones(in_features) * -10.0)
    
    def forward(self, x):
        return x + sample(self.mu, self.log_var)
    
class TemperatureScale(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mu = nn.Parameter(torch.ones(1))
        self.log_var = nn.Parameter(torch.ones(1) * -10.0)
 
    def forward(self, x):
        scaler = sample(self.mu, self.log_var)
        return torch.log(1.0 + torch.exp(scaler)) * x

class Decoder(nn.Module):
    def __init__(self, alphabet_size, convolution_depth, seq_len):
        super().__init__()

        self.stem = nn.Sequential(
            DecoderLayer(50, 300),  # 0, linear layer, weights are sampled from N(mu, sigma).
            nn.ReLU(),
            DecoderLayer(300, 1000),  # 2, linear layer with sampled weights.
            nn.ReLU(),
            DecoderLayer(1000, 2000),  # 4, linear layer with sampled weights.
            nn.ReLU(),
            DecoderLayer(2000, seq_len * convolution_depth, bias=False, xavier_init=True),  # 6, last hidden layer with sampled weights.
            nn.ReLU(),
            Rearrange('b (l c) -> (b l) c', c=convolution_depth),
            DecoderLayer(convolution_depth, alphabet_size, bias=False),  # 9, output convolution layer with sampled weights.
            Rearrange('(b l) a -> b (l a)', l=seq_len),
            AddBias(alphabet_size * seq_len),  # 11, add sampled bias to output.
            Rearrange('b (l a) -> b l a', a=alphabet_size),
            TemperatureScale(), # 13, temperature scaler is also sampled.
            # nn.LogSoftmax(dim=-1),
        )

    def forward(self, z):
        z = self.stem(z)
        return z

class EVE(nn.Module):
    def __init__(self, seq_len, alphabet_size=21, convolution_depth=40):
        super().__init__()
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size

        self.encoder = Encoder(alphabet_size, seq_len)
        self.decoder = Decoder(alphabet_size, convolution_depth, seq_len)
    
    def get_w_kl(self):
        kl_w = 0.0
        for layer_idx in [0, 2, 4, 6, 9]:
            for param_type in ['weight', 'bias']:
                if f'w_mu.{param_type}' not in self.decoder.stem[layer_idx].state_dict():
                    continue

                mu = self.decoder.stem[layer_idx].state_dict()[f'w_mu.{param_type}'].flatten()
                log_var = self.decoder.stem[layer_idx].state_dict()[f'w_log_var.{param_type}'].flatten()
                kl_w += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Add kl_w for last bias layer.
        mu = self.decoder.stem[11].state_dict()['mu'].flatten()
        log_var = self.decoder.stem[11].state_dict()['log_var'].flatten()
        kl_w += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Add kl_w for temperature scaler.
        mu = self.decoder.stem[13].state_dict()['mu'].flatten()
        log_var = self.decoder.stem[13].state_dict()['log_var'].flatten()
        kl_w += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return kl_w
    
    def forward(self, x, return_latent=False):
        z_mu, z_log_var = self.encoder(x)
        z = sample(z_mu, z_log_var)
        x = self.decoder(z)

        if return_latent:
            return x, z_mu, z_log_var
        else:
            return x
    
    def compute_evolutionary_index(self, wt_seq, mut_seq, num_samples=20_000):
        wt_seq, mut_seq = map(lambda x: rearrange(x, '... -> () ...'), [wt_seq, mut_seq])

        wt_elbos = []
        for _ in range(num_samples):
            seq_recon, z_mu, z_log_var = self.forward(wt_seq, return_latent=True)

            ce_loss = F.cross_entropy(
                seq_recon.view(-1, self.alphabet_size),
                wt_seq.argmax(dim=1).flatten(),
                reduction='sum'
            )
            z_kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
            w_kl_loss = self.get_w_kl()

            elbo = -1.0 * (ce_loss + z_kl_loss + w_kl_loss)
            wt_elbos.append(elbo.cpu().numpy())
        wt_elbos = np.array(wt_elbos)

        mut_elbos = []
        for _ in range(num_samples):
            seq_recon, z_mu, z_log_var = self.forward(mut_seq, return_latent=True)

            ce_loss = F.cross_entropy(
                seq_recon.view(-1, self.alphabet_size),
                mut_seq.argmax(dim=1).flatten(),
                reduction='sum'
            )
            z_kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
            w_kl_loss = self.get_w_kl()

            elbo = -1.0 * (ce_loss + z_kl_loss + w_kl_loss)
            mut_elbos.append(elbo.cpu().numpy())
        mut_elbos = np.array(mut_elbos)
        
        return np.mean(wt_elbos - mut_elbos)

if __name__ == '__main__':
    from Bio import SeqIO

    def get_sequence_length(a2m_fp):
        """Get the sequence length of the first sequence in the a2m file.
        a2m_fp: Path to a2m file.
        """
        for record in SeqIO.parse(a2m_fp, "fasta"):
            return len(record.seq)
        
    a2i = {a:i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY-')}
    def one_hot_encode_amino_acid(sequence):
        return torch.eye(len(a2i))[[a2i[a] for a in sequence]].T
        
    model = EVE(seq_len=100).cuda()

    msa = 'data/P53_HUMAN_b01.filtered.a2m'
    ALPHABET_SIZE = 21

    print('Loading pretrained model.')
    model = EVE(seq_len=get_sequence_length(msa), alphabet_size=ALPHABET_SIZE)
    model.load_state_dict(torch.load('ckpts/TP53.best.pt'))
    model.cuda()

    wt_seq = """LSPDDIEQWFTEDPGDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQG
SYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQ
SQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEV
GSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTE
EENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELN
EALELKDAQAGKEPGGSRAHSSHLKSKKG""".replace('\n', '')

    # F109Q
    mut_seq_pathogenic1 = """LSPDDIEQWFTEDPGDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQG
SYGQRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQ
SQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEV
GSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTE
EENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELN
EALELKDAQAGKEPGGSRAHSSHLKSKKG""".replace('\n', '')
    
    # V173Y
    mut_seq_pathogenic2 = """LSPDDIEQWFTEDPGDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQG
SYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQ
SQHMTEVYRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEV
GSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTE
EENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELN
EALELKDAQAGKEPGGSRAHSSHLKSKKG""".replace('\n', '')

    # M66V
    mut_seq_benign = """LSPDDIEQWFTEDPGDEAPRVPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQG
SYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQ
SQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEV
GSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTE
EENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELN
EALELKDAQAGKEPGGSRAHSSHLKSKKG""".replace('\n', '')

    wt_seq, mut_seq_benign, mut_seq_pathogenic1, mut_seq_pathogenic2 = map(
        lambda x: one_hot_encode_amino_acid(x).cuda(),
        [wt_seq, mut_seq_benign, mut_seq_pathogenic1, mut_seq_pathogenic2]
    )

    model.eval()
    with torch.no_grad():
        print(model.compute_evolutionary_index(wt_seq, mut_seq_pathogenic1))
        print(model.compute_evolutionary_index(wt_seq, mut_seq_pathogenic2))
        print(model.compute_evolutionary_index(wt_seq, mut_seq_benign))

