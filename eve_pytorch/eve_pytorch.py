import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SampledLinear(nn.Module):
    def __init__(self, in_features, out_features):
        self.to_mu = nn.Linear(in_features, out_features)
        self.to_log_var = nn.Linear(in_features, out_features)

        # nn.init.constant_(self.to_mu.bias, 0.1)
        # nn.init.constant_(self.to_z_log_var.bias, -10.0)
    
    def forward(self, x):
        mu = self.to_mu(x)
        log_var = self.to_log_var(x)
        return sample(mu, log_var)

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

        # Not now
        nn.init.constant_(self.to_z_mu.bias, 0.1)
        nn.init.constant_(self.to_z_log_var.bias, -10.0)

    def forward(self, x):
        x = rearrange(x, 'b c l -> b (c l)')

        x = self.stem(x)
        z_mu = self.to_z_mu(x)
        z_log_var = self.to_z_mu(x)

        return z_mu, z_log_var

class DecoderLayer(nn.Module):
    def __init__(self, in_features, out_features, use_xavier_init=False):
        super().__init__()

        self.w_mu = nn.Linear(in_features, out_features)
        self.w_log_var = nn.Linear(in_features, out_features)

        if use_xavier_init:
            nn.init.xavier_normal_(self.w_mu)
        else:
            nn.init.constant_(self.w_mu.weight, 0.1)
            nn.init.constant_(self.w_mu.bias, 0.1)

        nn.init.constant_(self.w_log_var.weight, -10.0)
        nn.init.constant_(self.w_log_var.bias, -10.0)

    def forward(self, z):
        w = sample(self.w_mu.weight, self.w_log_var.weight)
        b = sample(self.w_mu.bias, self.w_log_var.bias)

        return F.linear(z, weight=w, bias=b)

class Decoder(nn.Module):
    def __init__(self, alphabet_size, seq_len):
        super().__init__()

        self.stem = nn.Sequential(
            DecoderLayer(50, 300),
            nn.ReLU(),
            DecoderLayer(300, 1000),
            nn.ReLU(),
            DecoderLayer(1000, 2000),
            nn.ReLU(),
            DecoderLayer(2000, alphabet_size * seq_len),
            nn.ReLU(),
            Rearrange('b (l c) -> (b l) c', c=alphabet_size),
            DecoderLayer(alphabet_size, alphabet_size),
            Rearrange('(b l) c -> b c l', l=seq_len),
        )

    def forward(self, z):
        z = self.stem(z)
        return z

class EVE(nn.Module):
    def __init__(self, alphabet_size=20, seq_len=100):
        super().__init__()

        self.enc = Encoder(alphabet_size, seq_len)
        self.dec = Decoder(alphabet_size, seq_len)
    
    def forward(self, x):
        z_mu, z_log_var = self.enc(x)
        z = sample(z_mu, z_log_var)
        x = self.dec(z)
        return x

if __name__ == '__main__':
    model = EVE().cuda()

    x = torch.randn([16, 20, 100]).cuda()
    print(model(x).shape)
    