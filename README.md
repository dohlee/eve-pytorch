# eve-pytorch

![model](img/banner.png)

Implementation of evolutionary model of variant effect (EVE), a deep generative model of evolutionary data, in PyTorch. It's just an re-implementation of the official model for my own learning purpose, which is also implemented in PyTorch. The official implementation can be found [here](https://github.com/OATML-Markslab/EVE).

## Installation
```bash
$ pip install eve-pytorch
```

## Usage
```python
import torch
from eve_pytorch import EVE

SEQ_LEN = 1000
ALPHABET_SIZE = 21

model = EVE(seq_len=SEQ_LEN, alphabet_size=ALPHABET_SIZE)

# ... training ...

x = torch.randn(1, 4, 1000)

# If you want to get the reconstructed sequence only,
x_reconstructed = model(x, return_latent=False)

# or, if you want to get the latent variables
x_reconstructed, z_mu, z_log_var = model(x, return_latent=True)
```

## Training
```bash
$ python -m eve_pytorch.train \
  --msa data/msa.filtered.a2m  \ # Multiple sequence alignment.
  --output ckpts/best_checkpoint.pt \
  --use-wandb  # Optional, for logging
```

## Computing evolutionary index
The authors defined an **evolutionary index** of a protein variant as the relative fitness of mutated sequence $\mathbf{s}$ compared with that of a wildtype sequence $\mathbf{w}$. Since computing the exact log-likelihood is intractable, the authors approximated the log-likelihood ratio of the two sequences as the difference between the ELBO of the two sequences:

$$ELBO(\mathbf{w}) - ELBO(\mathbf{s})$$

In this reproduction, I implemented `EVE.compute_evolutionary_index()` method to compute the evolutionary index of a protein variant. The method takes two sequences as input, and returns the evolutionary index of the variant. Optionally, you can tweak `num_samples` parameter to control the number of samples for the Monte Carlo sampling of latent vectors.

```python
import torch
from eve_pytorch import EVE

SEQ_LEN = 1000
ALPHABET_SIZE = 21

model = EVE(seq_len=SEQ_LEN, alphabet_size=ALPHABET_SIZE)
model.load_state_dict(torch.load('path/to/best/checkpoint.pt'))

wt_seq = # One-hot encoded wildtype amino acid sequence.
mut_seq = # One-hot encoded mutated amino acid sequence.

model.eval()
with torch.no_grad():
  model.compute_evolutionary_index(wt_seq, mut_seq, num_samples=20_000)
```

## Citations

```bibtex
@article{frazer2021disease,
  title={Disease variant prediction with deep generative models of evolutionary data},
  author={Frazer, Jonathan and Notin, Pascal and Dias, Mafalda and Gomez, Aidan and Min, Joseph K and Brock, Kelly and Gal, Yarin and Marks, Debora S},
  journal={Nature},
  volume={599},
  number={7883},
  pages={91--95},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
```