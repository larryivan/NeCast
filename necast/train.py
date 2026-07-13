"""Train the network on simulated windows, with the recipe of the published model.

    necast train SIMULATIONS --out FOLDER

SIMULATIONS is a folder written by `necast simulate`, with windows of 30 Mb
and at least 32 individuals. Training has two steps.

Prepare: for every window, the 32 individuals NeCast reads, its summaries and
its PSMC+ posterior, saved as FOLDER/windows/<window>.npz. This is most of the
work, done once; a stopped run skips the windows already prepared.

Train: 500 windows (a tenth if there are fewer than 5,000) are kept aside to
choose the best epoch; the summaries of the others set the normalisation
constants. Then 12 epochs: in each, the windows in a new random order, one
window per forward pass, one AdamW update per 32 windows, the learning rate
following a cosine from 3e-4 to 1.5e-5 over the epochs. After each epoch the
error on the windows kept aside is measured, and the best network is saved as
FOLDER/necast.pt, for `necast run --weights`.
"""

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from necast import network, posterior, summaries
from necast.window import INDIVIDUALS, WINDOW_BP, InputError, Window

SEED = 20260728
EPOCHS = 12
BLOCK = 32                       # windows per update
LEARNING_RATE, LAST_RATE, WEIGHT_DECAY, CLIP = 3e-4, 1.5e-5, 1e-4, 1.0
VALIDATION = 500


# --- simulated windows -------------------------------------------------------

def read_simulations(folder):
    """Yield (window name, Window, true log10 Ne) for every window in a folder of `necast simulate`."""
    samples = Path(folder) / "samples"
    files = sorted(samples.glob("shard_*.npz")) or sorted(samples.glob("*fragments/fragment_*.npz"))
    if not files:
        raise InputError(f"no simulated windows in {samples}; make them with necast simulate")
    for path in files:
        z = np.load(path)
        rec_rates, mut_rates = (rate_offsets(z[f"obs_{m}_offsets"]) for m in ("rec", "mut"))
        for k, name in enumerate(z["sample_id"]):
            if int(z["sequence_length"][k]) != WINDOW_BP:
                raise InputError(f"{path}: window {name} is not {WINDOW_BP:,} bp long; NeCast trains on windows "
                                 "of 30 Mb (necast simulate --preset full_v01)")
            v0, v1 = z["variant_offsets"][k: k + 2]
            m0, m1 = z["missing_offsets"][k: k + 2]
            states = six_states(z["genotype_packed"][v0:v1], z["missing_flat_idx"][m0:m1].astype(np.int64),
                                int(z["n_haplotypes"][0]))
            maps = []
            for m, rates in (("rec", rec_rates), ("mut", mut_rates)):
                p0, p1 = z[f"obs_{m}_offsets"][k: k + 2]
                maps.append((z[f"obs_{m}_pos"][p0:p1], z[f"obs_{m}_rate"][rates[k]: rates[k + 1]]))
            yield str(name), Window("sim", np.array([[0, WINDOW_BP]]), z["variant_positions_bp"][v0:v1].astype(np.int64),
                                    np.ascontiguousarray(states[:, individuals(str(name), states.shape[1])]), *maps), \
                z["target_log10_ne"][k]


def six_states(packed, missing, n_haplotypes):
    """Genotype states of every individual (see necast.summaries) from packed haplotypes."""
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :n_haplotypes]
    lost = np.zeros(bits.shape, bool)
    lost.reshape(-1)[missing] = True
    called = 2 - lost[:, 0::2] - lost[:, 1::2].astype(int)
    derived = (bits[:, 0::2] & ~lost[:, 0::2]) + (bits[:, 1::2] & ~lost[:, 1::2])
    return np.where(called == 1, 3 + derived, np.where(called == 2, derived, 5)).astype(np.uint8)


def individuals(name, n):
    """The 32 of a window's n individuals NeCast reads: a fixed random choice per window."""
    if n < INDIVIDUALS:
        raise InputError(f"window {name} has {n} individuals; NeCast reads {INDIVIDUALS}")
    digest = hashlib.blake2b(f"dlcoalsim-ladder-v1:0:{name}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big")).permutation(n)[:INDIVIDUALS]


def rate_offsets(offsets):
    """Where each window's rates start: a map with b breakpoints has b - 1 rates."""
    return np.append(0, np.cumsum(np.maximum(np.diff(offsets.astype(np.int64)) - 1, 0)))


# --- prepare -----------------------------------------------------------------

def prepare(simulations, out, threads=1, log=print):
    """Save what training reads of each window: its genotypes at the polymorphic
    sites, maps, summaries, posterior and true history."""
    folder = Path(out) / "windows"
    folder.mkdir(parents=True, exist_ok=True)
    for name, w, target in read_simulations(simulations):
        path = folder / f"{name}.npz"
        if path.exists():
            continue
        started = time.perf_counter()
        args = (w.positions, w.states, w.rec_map, w.mut_map, WINDOW_BP)
        s = summaries.summarise(*args, threads=threads)
        p = posterior.summarise(*args, threads=threads)
        sites = summaries.segregating(w.states)
        temporary = folder / f"{name}.part.npz"
        np.savez_compressed(temporary, positions=w.positions[sites], states=w.states[sites],
                            rec_breaks=w.rec_map[0], rec_rates=w.rec_map[1],
                            mut_breaks=w.mut_map[0], mut_rates=w.mut_map[1], target=target,
                            sfs=s.sfs, ld=s.ld, ibs=s.ibs, ibs_exposure=s.ibs_exposure,
                            log10_watterson=np.float32(s.log10_watterson),
                            posterior=np.full(256, np.nan, np.float32) if p is None else p)
        temporary.rename(path)
        log(f"prepared {name}: {int(sites.sum()):,} sites, {time.perf_counter() - started:.0f} s")


class Prepared:
    """The prepared windows: summaries in memory, genotypes read from disk when needed."""

    SMALL = ("target", "sfs", "ld", "ibs", "ibs_exposure", "log10_watterson", "posterior")

    def __init__(self, out):
        self.files = sorted(f for f in (Path(out) / "windows").glob("*.npz") if not f.name.endswith(".part.npz"))
        columns = {key: [] for key in self.SMALL}
        for path in self.files:
            with np.load(path) as z:
                for key in self.SMALL:
                    columns[key].append(z[key])
        for key, values in columns.items():
            setattr(self, key, np.stack(values))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        """(Window, Summaries, posterior or None, true log10 Ne) of window i."""
        with np.load(self.files[i]) as z:
            w = Window("sim", np.array([[0, WINDOW_BP]]), z["positions"], z["states"],
                       (z["rec_breaks"], z["rec_rates"]), (z["mut_breaks"], z["mut_rates"]))
        s = summaries.Summaries(self.sfs[i], self.ld[i], self.ibs[i], self.ibs_exposure[i],
                                float(self.log10_watterson[i]))
        p = None if np.isnan(self.posterior[i]).any() else self.posterior[i]
        return w, s, p, self.target[i]

    def split(self):
        """(fit, validation): the validation windows are a fixed random choice."""
        keys = [hashlib.blake2b(f"necast-validation:{f.stem}".encode(), digest_size=8).hexdigest() for f in self.files]
        order = np.argsort(keys, kind="stable")
        n = min(VALIDATION, len(self) // 10)
        return np.sort(order[n:]), np.sort(order[:n])

    def constants(self, rows):
        """Normalisation constants and baseline offset from the windows `rows`, as in training."""
        exposure = np.broadcast_to(np.log1p(self.ibs_exposure[rows])[:, None, :, None], (len(rows), 1, 31, 27))
        streams = {"sfs": (np.log1p(self.sfs[rows]), 0),
                   "ld": (self.ld[rows].astype(np.float64), (0, 2)),
                   "ibs": (np.concatenate((np.log1p(self.ibs[rows].astype(np.float64)), exposure), 1), (0, 2))}
        out = {}
        for name, (x, axes) in streams.items():
            mean, std = x.mean(axes, keepdims=True)[0], x.std(axes, keepdims=True)[0]
            out[f"{name}_mean"] = mean.astype(np.float32)
            out[f"{name}_std"] = np.where(std <= 1e-12, 1.0, std).astype(np.float32)
        present = self.posterior[rows][~np.isnan(self.posterior[rows]).any(1)].astype(np.float64)
        std = present.std(0)
        out["posterior_mean"], out["posterior_std"] = present.mean(0), np.where(std > 0, std, 1.0)
        difference = self.target[rows] - self.log10_watterson[rows, None]
        out["offset"] = difference.mean(0, dtype=np.float64).astype(np.float32)
        return out


# --- train -------------------------------------------------------------------

def train(out, epochs=EPOCHS, device=None, log=print):
    """Train a network on the prepared windows in `out`; save the best as out/necast.pt."""
    out = Path(out)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    data = Prepared(out)
    if len(data) < 10:
        raise InputError(f"training needs at least 10 prepared windows, and {out / 'windows'} has {len(data)}")
    fit, validation = data.split()
    log(f"{len(fit):,} windows to fit, {len(validation):,} to choose the best epoch, on {device}")

    torch.manual_seed(SEED)
    net = network.Network()
    initialise(net)
    for name, value in data.constants(fit).items():
        getattr(net, name).copy_(torch.from_numpy(value))
    net.to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=LAST_RATE)

    best, history = math.inf, out / "training.tsv"
    history.write_text("epoch\tlearning_rate\tfit_rmse\tvalidation_rmse\tseconds\n")
    for epoch in range(1, epochs + 1):
        started, rate = time.perf_counter(), optimiser.param_groups[0]["lr"]
        net.train()
        order, squares = np.random.default_rng(SEED + epoch).permutation(fit), 0.0
        for start in range(0, len(order), BLOCK):
            optimiser.zero_grad(set_to_none=True)
            for i in order[start: start + BLOCK]:
                w, s, p, target = data[i]
                loss = F.mse_loss(net(w, s, p), torch.from_numpy(target).to(device))
                (loss / BLOCK).backward()            # the last, shorter block too: one scale for every window
                squares += loss.item()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            optimiser.step()
        schedule.step()
        rmse = validation_rmse(net, data, validation)
        seconds = time.perf_counter() - started
        with open(history, "a") as f:
            f.write(f"{epoch}\t{rate:.6g}\t{math.sqrt(squares / len(fit)):.6f}\t{rmse:.6f}\t{seconds:.0f}\n")
        log(f"epoch {epoch}: validation RMSE {rmse:.4f}, {seconds:.0f} s")
        if rmse < best:
            best, state = rmse, copy.deepcopy(net.state_dict())
    torch.save(state, out / "necast.pt")
    log(f"saved the network of the best epoch (validation RMSE {best:.4f}) as {out / 'necast.pt'}")


@torch.no_grad()
def validation_rmse(net, data, rows):
    """Root mean square error of log10 Ne over the windows and their 64 intervals."""
    net.eval()
    squares = 0.0
    for i in rows:
        w, s, p, target = data[i]
        squares += float(np.sum((net(w, s, p).cpu().numpy().astype(np.float64) - target) ** 2))
    return math.sqrt(squares / (64 * len(rows)))


def initialise(net):
    """The published model's starting point for the few parameters PyTorch's
    defaults do not cover: small random queries and embeddings, and a time
    bias that points each interval at the resolution whose genetic length,
    at 1e-8 per bp, is closest to 1/(2t) Morgans for the interval's time t."""
    torch.nn.init.normal_(net.time_query, std=0.02)
    torch.nn.init.normal_(net.scale_embedding, std=0.02)
    torch.nn.init.normal_(net.summary_encoder.time_basis, std=0.02)
    t = torch.from_numpy(np.sqrt(posterior.GENERATIONS[:-1] * posterior.GENERATIONS[1:]))   # middle of each interval
    morgans = torch.tensor([network.REGION * 4.0 ** i for i in range(4)], dtype=torch.float64) * 1e-8
    octaves = torch.log2(morgans[None] * 2 * t[:, None])
    bias = -0.5 * octaves.square()
    with torch.no_grad():
        net.time_scale_bias.copy_(bias - bias.mean(1, keepdim=True))
