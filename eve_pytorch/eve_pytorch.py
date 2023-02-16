import torch
import torch.nn as nn

from einops import rearrange

# {   "encoder_parameters": {
#         "hidden_layers_sizes"         :   [2000,1000,300],
#         "z_dim"                               :   50,
#         "convolve_input"                      :   false,
#         "convolution_input_depth"             :   40,
#         "nonlinear_activation"                :   "relu",
#         "dropout_proba"                       :   0.0
#     },
#     "decoder_parameters": {
#         "hidden_layers_sizes"         :   [300,1000,2000],
#         "z_dim"                               :   50,
#         "bayesian_decoder"                    :   true,
#         "first_hidden_nonlinearity"           :   "relu", 
#         "last_hidden_nonlinearity"            :   "relu", 
#         "dropout_proba"                       :   0.1,
#         "convolve_output"                     :   true,
#         "convolution_output_depth"            :   40, 
#         "include_temperature_scaler"          :   true, 
#         "include_sparsity"                    :   false, 
#         "num_tiles_sparsity"                  :   0,
#         "logit_sparsity_p"                    :   0
#     },
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

class Encoder(nn.Module):
    def __init__(self, seq_len, alphabet_size):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(alphabet_size * seq_len, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 300),
            nn.ReLU(),
        )
        self.to_z_mean = nn.Linear(300, 50)
        # nn.init.constant_(self.to_z_mean.bias, 0.1)
        self.to_z_log_var = nn.Linear(300, 50)
        # nn.init.constant_(self.to_z_log_var.bias, -10.0)

    def forward(self, x):
        x = rearrange(x, 'b c l -> b (c l)')
        x = self.linear(x)

        z_mean = self.to_z_mean(x)
        z_log_var = self.to_z_log_var(x)

        return z_mean, z_log_var

class EVE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

if __name__ == '__main__':
    model = EVE()
    