"""The results of a run: two tables and a figure."""

import numpy as np

from necast.posterior import GENERATIONS

PURPLE = "#5242A7"


def write(out, results, generation_time=None, truth=None):
    """Write necast.tsv, windows.tsv and necast.png into the folder `out`.

    `results` holds one (chromosome, start, end, group, log10 Ne) per window
    and group of 32 individuals. The estimate is the mean of log10 Ne over all
    of them. Returns that mean."""
    curves = np.array([r[4] for r in results])
    mean = curves.mean(0)
    columns = ["interval", "start_generations", "end_generations"]
    rows = [[i + 1, GENERATIONS[i], GENERATIONS[i + 1]] for i in range(64)]
    if generation_time:
        columns += ["start_years", "end_years"]
        rows = [row + [row[1] * generation_time, row[2] * generation_time] for row in rows]
    table(out / "necast.tsv", columns + ["ne"], [row + [10 ** x] for row, x in zip(rows, mean)])
    table(out / "windows.tsv", ["chromosome", "start", "end", "group"] + [f"ne_{i + 1}" for i in range(64)],
          [[c, s, e, g, *10 ** x] for c, s, e, g, x in results])
    plot(out / "necast.png", curves, mean, generation_time, truth)
    return mean


def table(path, columns, rows):
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(x) if isinstance(x, (str, int)) else f"{x:.6g}" for x in row) + "\n")


def plot(path, curves, mean, generation_time=None, truth=None):
    """Each window's estimate in grey, their mean in purple, the truth (if known) in black."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = GENERATIONS * (generation_time or 1)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    if len(curves) > 1:
        for i, curve in enumerate(curves):
            ax.stairs(10 ** curve, edges, baseline=None, color="0.8", linewidth=0.6,
                      label="single windows" if i == 0 else None)
    if truth is not None:
        ax.stairs(10 ** truth, edges, baseline=None, color="black", linewidth=1.2, label="true history")
    ax.stairs(10 ** mean, edges, baseline=None, color=PURPLE, linewidth=2, label="NeCast")
    ax.set(xscale="log", yscale="log", xlim=(edges[0], edges[-1]), ylabel="Effective population size",
           xlabel=f"Years ago ({generation_time} years per generation)" if generation_time else "Generations ago")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)
