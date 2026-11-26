"""Draw the four figures in the README, one for each idea it shows.

Run `uv run compare/plot.py` to write them next to this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import (
    brier_variance,
    certain,
    confident,
    entropies,
    incoherence,
    one_odd_run,
    shifted_pair,
)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # options A to D, as in toy/plot.py
SERIES = [
    ("Incoherence", "#2a78d6"),
    ("Brier variance", "#eb6834"),
    ("Square root of Brier variance", "#1baf7a"),
]
BASES = np.linspace(0.54, 0.98, 45)  # share the first sample of a pair puts on option A
PAIRS = (("46% and 54%", 0.54, 3), ("90% and 98%", 0.98, 2))  # label, base, decimals
COUNTS = np.unique(np.logspace(1, 4, 31).astype(int))  # ensemble sizes, 10 to 10,000
HERE = Path(__file__).parent


def measures(p):
    """Return the three plotted measures for samples `p`."""
    brier = brier_variance(p)
    return incoherence(p), brier, np.sqrt(brier)


def tidy(ax):
    """Give an axis the shared look: no spines, no tick marks, muted labels."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(length=0, labelcolor=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def heading(ax, title, subtitle):
    """Put a bold title and a muted line under it above an axis."""
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10.5)


def series_legend(fig, x=0.5):
    """Name the three measure lines under the figure."""
    handles = [plt.Line2D([], [], color=colour, linewidth=2) for _, colour in SERIES]
    fig.legend(
        handles, [name for name, _ in SERIES], loc="lower center", bbox_to_anchor=(x, 0),
        ncol=3, frameon=False, fontsize=10.5, labelcolor=INK, columnspacing=1.4,
    )  # fmt: skip


def save(fig, name):
    """Write a figure next to this file and print where it went."""
    out = HERE / name
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def draw_lines(ax, x, ensembles):
    """Draw one line per measure across `ensembles`, with the value at each line's end."""
    values = np.array([measures(p) for p in ensembles])
    tidy(ax)
    for (_, colour), y in zip(SERIES, values.T):
        ax.plot(x, y, color=colour, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot(x[-1], y[-1], "o", color=colour, markersize=8, zorder=4, clip_on=False)
    # Label line ends from the top down, skipping any that would sit on the one above.
    last = np.inf
    for y in sorted(values[-1], reverse=True):
        if last - y > 0.045:
            ax.annotate(
                f"{y:.2f}", (x[-1], y), xytext=(9, 0), textcoords="offset points",
                va="center", color=INK, fontsize=10.5, annotation_clip=False,
            )  # fmt: skip
            last = y
    ax.set_ylim(0, 0.72)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def pool_figure():
    """Draw two sets of two samples as stacked bars, each with its pool underneath."""
    sets = [
        ("Two samples that agree", np.vstack([confident(0), confident(0)])),
        ("Two samples that disagree", np.vstack([confident(0), confident(1)])),
    ]
    rows = [2.1, 1.1, -0.3]  # the pool sits a little apart from the samples
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), facecolor=SURFACE)
    for ax, (title, p) in zip(axes, sets):
        each, pool = entropies(p)
        bars = np.vstack([p, p.mean(0)]) * 100
        left = np.zeros(3)
        for option, colour in enumerate(COLOURS):
            ax.barh(
                rows, bars[:, option], left=left, color=colour,
                height=0.7, edgecolor=SURFACE, linewidth=0.9,
            )  # fmt: skip
            left += bars[:, option]
        for y, entropy in zip(rows, (*each, pool)):
            ax.text(
                103, y, f"entropy {entropy:.2f}", va="center", fontsize=10.5,
                color=INK, fontweight="bold" if y < 0 else "normal",
            )  # fmt: skip
        tidy(ax)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.9, 2.7)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_yticks(rows)
        ax.set_yticklabels(["Sample 1", "Sample 2", "Their pool"], color=INK, fontsize=10.5)
        heading(
            ax, title,
            f"The gap is {pool - each[0]:.2f} and Incoherence is {incoherence(p):.2f}",
        )  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in COLOURS]
    fig.legend(
        handles, ["A", "B", "C", "D"], title="Probability given to each option",
        loc="lower center", ncol=4, frameon=False,
        fontsize=10.5, labelcolor=INK, title_fontsize=10.5,
    )  # fmt: skip
    fig.suptitle(
        "Samples that disagree pool into something more spread out",
        x=0.035, ha="left", color=INK, fontsize=15, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.035, 0.865,
        "Entropy is how spread out one sample is, from 0 when certain to 2 when even over four"
        " options",
        color=MUTED, fontsize=11,
    )  # fmt: skip
    fig.subplots_adjust(left=0.08, right=0.89, top=0.66, bottom=0.25, wspace=0.62)
    save(fig, "pool.png")


def draw_curve(ax):
    """Draw the entropy of a two option sample, and mark the two pairs on it."""
    share = np.linspace(0.001, 0.999, 400)
    ax.plot(
        share * 100, -share * np.log2(share) - (1 - share) * np.log2(1 - share),
        color=MUTED, linewidth=2, zorder=3,
    )  # fmt: skip
    tidy(ax)
    for label, base, decimals in PAIRS:
        p = shifted_pair(base)
        x, y = p[:, 0] * 100, entropies(p)[0]
        ax.axvspan(x.min(), x.max(), color=GRID, zorder=1)  # the same width both times
        ax.plot(
            x, y, "o", color=INK, markersize=8, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=2,
        )  # fmt: skip
        low, high = sorted(y)
        # Label the steep pair beside it and the flat pair underneath, inside the curve.
        steep = high - low > 0.1
        ax.text(
            x.min() - 3 if steep else x.mean(), y.mean() if steep else 0.8,
            f"{label}\nentropy {high:.{decimals}f} and {low:.{decimals}f}",
            ha="right" if steep else "center", va="center", color=INK, fontsize=10.5,
        )  # fmt: skip
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share the sample puts on option A", color=MUTED, fontsize=10.5)


def wobble_figure():
    """Draw why the same wobble reads differently, then what each measure makes of it."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=SURFACE)
    draw_curve(axes[0])
    heading(axes[0], "Entropy of one sample", "Each shaded band is 8 points wide")
    draw_lines(axes[1], BASES * 100, [shifted_pair(base) for base in BASES])
    heading(axes[1], "What each measure reads", "As the pair moves towards certainty")
    axes[1].set_xticks([60, 70, 80, 90])
    axes[1].set_xticklabels(["60%", "70%", "80%", "90%"])
    axes[1].set_xlabel("Share the first sample puts on option A", color=MUTED, fontsize=10.5)
    series_legend(fig, x=0.72)
    fig.suptitle(
        "The same wobble moves entropy a lot near certainty and hardly at all near 50%",
        x=0.045, ha="left", color=INK, fontsize=15, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.045, 0.9,
        "Two samples over two options, always 8 percentage points apart",
        color=MUTED, fontsize=11,
    )  # fmt: skip
    fig.subplots_adjust(left=0.05, right=0.95, top=0.74, bottom=0.2, wspace=0.2)
    save(fig, "wobble.png")


def odd_figure(name, title, subtitle, odd, usual):
    """Draw each measure as one odd sample sits among more and more usual ones."""
    fig, ax = plt.subplots(figsize=(7, 5.2), facecolor=SURFACE)
    draw_lines(ax, COUNTS, [one_odd_run(count, odd, usual) for count in COUNTS])
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_xlabel("Number of samples in the set, log scale", color=MUTED, fontsize=10.5)
    series_legend(fig)
    fig.suptitle(title, x=0.08, ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.08, 0.875, subtitle, color=MUTED, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.82, bottom=0.22)
    save(fig, name)


def main():
    """Draw and save all four figures."""
    pool_figure()
    wobble_figure()
    odd_figure(
        "odd-sample.png", "One even sample among certain ones",
        "It is even over four options, the rest are certain of A",
        np.full(4, 0.25), certain(0),
    )  # fmt: skip
    odd_figure(
        "other-option.png", "One sample that favours another option",
        "It is 97% on B, the rest are 97% on A",
        confident(1), confident(0),
    )  # fmt: skip


if __name__ == "__main__":
    main()
