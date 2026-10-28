"""Draws the two data figures of the documentation:  python docs/img/figures.py

It needs the simulator's packages (pip install -e ".[simulate]") and reads the
paper's 1000 Genomes estimates from Hugging Face.
"""

import dataclasses
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from necast import report
from necast.posterior import GENERATIONS
from necast.simulator.config import Config, apply_preset
from necast.simulator.demography import DEMO_PROB_FIELDS, sample_custom_demography

HERE = Path(__file__).parent
LIGHT_PURPLE = "#AEA6DC"
ESTIMATES = ("https://huggingface.co/datasets/Larrivhan/NeCast-1000G/resolve/main/"
             "populations_25/results/curves_bundle4.json")


def histories():
    """Three histories of each kind, drawn as `necast simulate --preset full_v01` draws them."""
    config = apply_preset(Config(preset="full_v01"))
    kinds = sorted(DEMO_PROB_FIELDS, key=lambda kind: -getattr(config, kind[1]))
    fig, axes = plt.subplots(3, 4, figsize=(10, 6), sharex=True, sharey=True, layout="constrained")
    for panel, (ax, (name, _)) in enumerate(zip(axes.flat, kinds)):
        only = dataclasses.replace(config, **{field: float(kind == name) for kind, field in DEMO_PROB_FIELDS})
        for draw, colour in enumerate((LIGHT_PURPLE, LIGHT_PURPLE, report.PURPLE)):
            log10_ne = sample_custom_demography(np.random.default_rng((panel, draw)), only)[1]
            ax.stairs(10 ** log10_ne, GENERATIONS, baseline=None, color=colour, linewidth=1.5)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set(xscale="log", yscale="log", xlim=(GENERATIONS[0], GENERATIONS[-1]), ylim=(70, 1.5e6))
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Generations ago", fontsize=10)
    fig.supylabel("Effective population size", fontsize=10)
    fig.savefig(HERE / "histories.png", dpi=200)


def yri():
    """The paper's estimate for the Yoruba of the 1000 Genomes Project, drawn as necast.png is."""
    with urllib.request.urlopen(ESTIMATES) as f:
        estimate = json.load(f)["populations"]["YRI"]["arms"]["d_both_maps"]
    report.plot(HERE / "yri.png", np.array(estimate["per_window_curves"]),
                np.array(estimate["mean_per_bin"]), generation_time=29)


if __name__ == "__main__":
    histories()
    yri()
