"""The NeCast network: one window in, log10 Ne on 64 time intervals out.

The network reads a window in two ways and adds the readings up.

Genotypes. Each polymorphic site gets an embedding per individual, from the
genotype and the site's place inside its 100-bp leaf. Embeddings are summed per
leaf and merged up a binary tree into 12.8-kb regions. Dilated convolutions
then run along the window, for each individual separately; attention among the
32 individuals and a pooling over them, which ignores their order, leave one
vector per region. Joined with the region's map exposure and site counts, the
regions are read at four resolutions by 64 learned queries, one per interval.

Summaries. The folded SFS, LD, IBS tracts and PSMC+ posterior are each encoded
into one vector per interval.

The five readings are projected, added and passed through a small head. Its
output corrects a baseline: Watterson's estimate of the window plus a fixed
offset per interval, the mean of truth minus Watterson's estimate over the
training windows.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from necast import posterior, summaries
from necast.window import INDIVIDUALS, WINDOW_BP

WEIGHTS = Path(__file__).parent / "weights" / "necast.pt"
LEAF = 100                                  # bp per leaf of the tree
LEVELS = 7                                  # leaves to regions: 100 bp x 2^7 = 12.8 kb
REGION = LEAF << LEVELS
REGIONS = math.ceil(WINDOW_BP / REGION)     # 2,344
WIDTH = 128
CALLS = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 0.0])         # alleles called, by genotype state
CONTEXT = torch.tensor([math.log10(WINDOW_BP), math.log2(2 * INDIVIDUALS)])   # window length, haplotypes

# constants fixed by the training windows: means and standard deviations of the
# summaries, and the offset of the baseline
CONSTANTS = {"sfs_mean": (16,), "sfs_std": (16,), "ld_mean": (5, 1), "ld_std": (5, 1),
             "ibs_mean": (8, 1, 27), "ibs_std": (8, 1, 27),
             "posterior_mean": (256,), "posterior_std": (256,), "offset": (64,)}


def load(path=WEIGHTS):
    """The trained network, ready to estimate."""
    network = Network()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.eval()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # sites to regions
        self.site_mlp = mlp(6 + 3, 8, 8)                # one-hot genotype, x, x^2, 1
        self.tree_left = nn.ModuleList(nn.Linear(8, 8, bias=False) for _ in range(LEVELS))
        self.tree_right = nn.ModuleList(nn.Linear(8, 8, bias=False) for _ in range(LEVELS))
        self.tree_norm = nn.ModuleList(nn.LayerNorm(8) for _ in range(LEVELS))
        # along the window, then across individuals
        self.to_individual = nn.Linear(8, 32, bias=False)
        self.individual_blocks = nn.ModuleList(ConvBlock(32, 2 ** i) for i in range(11))
        self.across = AcrossIndividuals(32)
        self.map_mlp = dense(8 + 2, 32)
        self.region_mlp = dense(2 * 32, WIDTH)
        self.region_blocks = nn.ModuleList(ConvBlock(WIDTH, dilation) for dilation in (1, 4, 16))
        # 64 queries over four resolutions
        self.downsample = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 5, stride=4, padding=2) for _ in range(3))
        self.scale_embedding = nn.Parameter(torch.zeros(4, WIDTH))
        self.memory_norm = nn.LayerNorm(WIDTH)
        self.memory_key = nn.Linear(WIDTH, WIDTH, bias=False)
        self.memory_value = nn.Linear(WIDTH, WIDTH, bias=False)
        self.time_query = nn.Parameter(torch.zeros(64, WIDTH))
        self.time_scale_bias = nn.Parameter(torch.zeros(64, 4))
        self.context_mlp = mlp(2, WIDTH, WIDTH)
        # summaries, then all five readings together
        self.summary_encoder = SummaryEncoder()
        self.fusion = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(5))
        self.fusion_norm = nn.LayerNorm(WIDTH)
        self.head_norm = nn.LayerNorm(WIDTH)
        self.head = mlp(WIDTH, WIDTH, 1)
        for name, shape in CONSTANTS.items():
            self.register_buffer(name, torch.zeros(shape, dtype=torch.float64 if "posterior" in name else torch.float32))

    @torch.no_grad()
    def estimate(self, window, threads=1):
        """log10 Ne of a window on the 64 intervals of necast.posterior.GENERATIONS."""
        args = (window.positions, window.states, window.rec_map, window.mut_map, WINDOW_BP)
        return self(window, summaries.summarise(*args, threads=threads),
                    posterior.summarise(*args, threads=threads)).cpu().numpy()

    def forward(self, window, s, p):
        """log10 Ne from a window, its summaries `s` and PSMC+ posterior `p` (None if missing)."""
        device = self.offset.device
        sites = summaries.segregating(window.states)
        positions = torch.from_numpy(window.positions[sites].astype(np.int64)).to(device)
        states = torch.from_numpy(window.states[sites]).to(device)
        exposure = map_exposure(window.rec_map, window.mut_map).to(device)
        readings = [self.read_genotypes(positions, states, exposure)]
        readings += self.summary_encoder(*(x.to(device) for x in self.normalise(s, p)))
        fused = self.fusion_norm(sum(project(x) for project, x in zip(self.fusion, readings)))
        return self.head(self.head_norm(fused)).squeeze(-1) + self.offset + s.log10_watterson

    def normalise(self, s, p):
        """The summaries as the network reads them: logs of counts, centred and
        scaled by the training windows' moments, in float64 as in training.
        A missing posterior is replaced by the training mean."""
        c = {name: buffer.cpu().numpy() for name, buffer in self.named_buffers()}
        exposure = np.broadcast_to(np.log1p(s.ibs_exposure)[None, :, None], (1, 31, 27))
        inputs = {"sfs": np.log1p(s.sfs), "ld": s.ld.astype(np.float64),
                  "ibs": np.concatenate((np.log1p(s.ibs.astype(np.float64)), exposure)),
                  "posterior": c["posterior_mean"] if p is None else p.astype(np.float64)}
        return [torch.from_numpy(((x - c[f"{name}_mean"]) / c[f"{name}_std"]).astype(np.float32))
                for name, x in inputs.items()]

    def read_genotypes(self, positions, states, exposure):
        """Genotypes and maps -> [64, WIDTH], one vector per time interval."""
        regions, sites, calls = self.regions(positions, states)
        x = self.to_individual(regions).permute(1, 2, 0)       # [individuals, channels, regions]
        for block in self.individual_blocks:                   # reach: 2,047 regions either side
            x = block(x)
        tokens = self.across(x.permute(2, 0, 1))               # [regions, 32]
        called = torch.where(sites > 0, calls / (2.0 * INDIVIDUALS * sites.clamp_min(1.0)), 0.0)
        features = torch.cat((exposure, torch.log1p(sites)[:, None], called[:, None]), 1)
        x = self.region_mlp(torch.cat((tokens, self.map_mlp(features)), 1)).T[None]   # [1, WIDTH, regions]
        for block in self.region_blocks:
            x = block(x)
        memory, scale = [], []
        for level in range(4):                                 # 12.8, 51.2, 204.8 and 819.2 kb per vector
            memory.append(x[0].T + self.scale_embedding[level])
            scale.append(torch.full((x.shape[-1],), level, device=x.device))
            if level < 3:
                x = F.gelu(self.downsample[level](x))
        memory, scale = self.memory_norm(torch.cat(memory)), torch.cat(scale)
        logits = self.time_query @ self.memory_key(memory).T / math.sqrt(WIDTH) + self.time_scale_bias[:, scale]
        return torch.softmax(logits, 1) @ self.memory_value(memory) + self.context_mlp(CONTEXT.to(x.device))

    def regions(self, positions, states):
        """Site embeddings summed into regions: [REGIONS, 32, 8], with the
        number of sites and of called alleles in each region."""
        n, device = len(positions), positions.device
        states = states.long()
        x = (positions % LEAF).float() / LEAF * 2.0 - 1.0     # place inside the leaf, in [-1, 1)
        inputs = torch.cat((torch.eye(6, device=device).expand(n, 6, 6),
                            torch.stack((x, x.square(), torch.ones_like(x)), 1)[:, None].expand(n, 6, 3)), 2)
        table = self.site_mlp(inputs)                          # [n, 6, 8]: the site in each genotype state
        ids, values, count = group_sum(positions // LEAF, table[torch.arange(n, device=device)[:, None], states])
        values = values / count.float().sqrt()[:, None, None]
        for left, right, norm in zip(self.tree_left, self.tree_right, self.tree_norm):
            values = torch.where((ids % 2 == 1)[:, None, None], right(values), left(values))
            ids, values, _ = group_sum(ids // 2, values)
            values = F.gelu(norm(values))
        regions = values.new_zeros(REGIONS, INDIVIDUALS, values.shape[-1])
        regions[ids] = values                                  # regions without sites stay exactly zero
        region = positions // REGION
        sites = torch.bincount(region, minlength=REGIONS).float()
        calls = torch.zeros(REGIONS, device=device).index_add_(0, region, CALLS.to(device)[states].sum(1))
        return regions, sites, calls


class AcrossIndividuals(nn.Module):
    """Attention among the individuals of each region, then their mean and mean
    square. The result does not depend on the order of the individuals."""

    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.norm_in, self.norm_attention, self.norm_ffn = (nn.LayerNorm(width) for _ in range(3))
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.ffn = mlp(width, 2 * width, width)
        self.pool = dense(2 * width, width)

    def forward(self, x):                                      # [regions, individuals, width] -> [regions, width]
        r, d, w = x.shape
        q, k, v = (t.reshape(r, d, self.heads, -1).transpose(1, 2) for t in self.qkv(self.norm_in(x)).chunk(3, -1))
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(w // self.heads), -1) @ v
        x = self.norm_attention(x + self.attention_out(a.transpose(1, 2).reshape(r, d, w)))
        x = x + self.ffn(self.norm_ffn(x))
        return self.pool(torch.cat((x.mean(1), x.square().mean(1)), 1))


class SummaryEncoder(nn.Module):
    """Folded SFS, LD, IBS tracts and PSMC+ posterior -> four [64, WIDTH] readings."""

    def __init__(self):
        super().__init__()
        self.context_mlp = mlp(2, WIDTH, WIDTH)
        self.sfs = nn.Sequential(dense(16, WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU())
        self.ld_convs = convs(5 + 1)
        self.ld_pool = dense(2 * WIDTH, WIDTH)
        self.ibs_convs = convs(8 + 1)
        self.ibs_region = dense(2 * WIDTH, WIDTH)
        self.ibs_pyramid = dense(10 * WIDTH, WIDTH)
        self.posterior = nn.Sequential(dense(256, WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU())
        self.time_basis = nn.Parameter(torch.zeros(4, 64, WIDTH))
        self.time_norm = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in range(4))

    def forward(self, sfs, ld, ibs, p):
        # the summaries' context (window length and haplotypes) is the same for
        # every window, so after normalisation it is zero
        c = self.context_mlp(torch.zeros(2, device=sfs.device))
        readings = (self.sfs(sfs), self.read_ld(ld), self.read_ibs(ibs), self.posterior(p))
        return [norm(x + c + basis) for norm, x, basis in zip(self.time_norm, readings, self.time_basis)]

    def read_ld(self, ld):                                     # [5, 26]: channels x distance classes
        coordinate = torch.linspace(-1.0, 1.0, 26, device=ld.device)
        h = self.ld_convs(torch.cat((ld, coordinate[None]))[None])
        return self.ld_pool(torch.cat((h.mean(2), h.amax(2)), 1))[0]

    def read_ibs(self, ibs):                                   # [8, 31, 27]: channels x regions x length classes
        coordinate = torch.linspace(-1.0, 1.0, 27, device=ibs.device).expand(31, 1, 27)
        h = self.ibs_convs(torch.cat((ibs.transpose(0, 1), coordinate), 1))
        regions = self.ibs_region(torch.cat((h.mean(2), h.amax(2)), 1))         # [31, WIDTH]
        pooled = [f(part, 0) for part in regions.split(summaries.REGION_SIZES) for f in (torch.mean, torch.amax)]
        return self.ibs_pyramid(torch.cat(pooled))


class ConvBlock(nn.Module):
    """x + pointwise(GELU(dilated depthwise convolution(GroupNorm(x))))."""

    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.depthwise = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width, bias=False)
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + self.pointwise(F.gelu(self.depthwise(self.norm(x))))


def mlp(n_in, n_hidden, n_out):
    """Linear, GELU, Linear."""
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))


def dense(n_in, n_out):
    """Linear, GELU, LayerNorm."""
    return nn.Sequential(nn.Linear(n_in, n_out), nn.GELU(), nn.LayerNorm(n_out))


def convs(n_in):
    """Two convolutions of width 3 along the classes, each followed by GELU."""
    return nn.Sequential(nn.Conv1d(n_in, WIDTH, 3, padding=1), nn.GELU(),
                         nn.Conv1d(WIDTH, WIDTH, 3, padding=1), nn.GELU())


def group_sum(keys, values):
    """Sum the rows of `values` whose sorted `keys` are equal: keys, sums and rows per key."""
    ids, inverse, counts = torch.unique_consecutive(keys, return_inverse=True, return_counts=True)
    return ids, values.new_zeros(len(ids), *values.shape[1:]).index_add_(0, inverse, values), counts


def map_exposure(rec_map, mut_map):
    """Eight numbers per region: its place in the window; then for each map the
    rate summed up to the region's centre (as is, and relative to the whole
    window) and inside the region. In float32, as in training."""
    edges = (torch.arange(REGIONS + 1, dtype=torch.float32) * REGION).clamp(max=WINDOW_BP)
    centres = 0.5 * (edges[:-1] + edges[1:])
    columns = [centres / WINDOW_BP, (edges[1:] - edges[:-1]) / WINDOW_BP]
    x = torch.cat((edges, centres))
    for breaks, rates in (rec_map, mut_map):
        breaks, rates = torch.tensor(breaks, dtype=torch.float32), torch.tensor(rates, dtype=torch.float32)
        area = torch.cat((torch.zeros(1), torch.cumsum((breaks[1:] - breaks[:-1]) * rates, 0)))
        i = (torch.searchsorted(breaks, x, right=True) - 1).clamp(0, len(rates) - 1)
        at_edges, at_centres = (area[i] + (x - breaks[i]) * rates[i]).split((REGIONS + 1, REGIONS))
        columns += [torch.log1p(at_centres * 1e6), at_centres / at_edges[-1],
                    torch.log1p((at_edges[1:] - at_edges[:-1]) * 1e6)]
    return torch.stack(columns, 1)
