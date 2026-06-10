# Training the network

NeCast comes with the network of the paper, so analysing data needs no
training. Train a network of your own only to change what it learns from:
other histories, other maps, other error rates. The window length (30 Mb) and
the number of individuals read (32) are fixed.

```bash
pip install "necast[simulate]"
necast simulate --preset full_v01 --n-samples 20000 --out-dir simulations
necast train simulations --out trained
necast run data.vcf.gz --mask accessible.bed --mu 1.25e-8 --rec-rate 1e-8 --out results --weights trained/necast.pt
```

The windows must be 30 Mb long and hold at least 32 individuals (`full_v01`
has 64); NeCast reads 32 of them, the same 32 every time.

## What `necast train` does

**Prepare.** For every window it saves, in `trained/windows/`, the
genotypes of its 32 individuals at the polymorphic sites, its maps, its
summaries and its PSMC+ posterior. This is the slow part: a few seconds a
window with 16 CPU threads, and about 1 MB of disk. A run that stops can be
started again and skips the windows already done. `--prepare-only` stops
after this step, so it can run on a machine without a GPU.

**Train.** 500 windows (a tenth, with fewer than 5,000 windows) are kept
aside to pick the best epoch; the others are fitted. The recipe is the one
the published network was trained with:

- the summaries are normalised with the means and standard deviations of the
  fitted windows;
- 12 epochs, each visiting the windows in a new random order (seeded);
- one window per forward pass, an AdamW update (weight decay 1e-4) every 32
  windows, gradients clipped to a norm of 1;
- a learning rate falling along a cosine from 3e-4 to 1.5e-5, one step per
  epoch;
- the loss is the mean squared error of log10 Ne over the 64 intervals.

A GPU is used when there is one (`--device` chooses). The published network
was trained on 89,996 windows for about 50 hours on four RTX 4090 GPUs, which
shared the windows of each update; `necast train` uses one GPU. A new run does
not reproduce the published weights exactly, because the random starting
values depend on the PyTorch version and GPUs round differently.

## What it writes

| File | Contents |
|---|---|
| `necast.pt` | the network of the epoch with the lowest validation error, for `necast run --weights` |
| `training.tsv` | per epoch: learning rate, error on the fitted windows, error on the windows kept aside, seconds |
| `windows/` | the prepared windows |
| `log.txt` | what was done |

The errors are root mean squared errors of log10 Ne. The published network
reached 0.296 on its 500 validation windows.

## The paper's training data

The 109,248 simulated windows behind the published network, with their
summaries and the splits used, are at
https://huggingface.co/datasets/Larrivhan/NeCast-simulations. They were made by
the simulator at git tag `corpus-generator`; the current simulator gives the
same windows except those whose maps are slices of the real maps (a fifth).
