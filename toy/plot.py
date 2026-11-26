"""Draw the toy example: 30 samples per panel, each a probability vector over A to D.

Run `uv run toy/plot.py` to write `toy/figure.png`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import OPTIONS, SAMPLES, build, measure, shapes

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PANELS = [
    ("Every sample has its own mix", "own mix (cloud)"),
    ("Samples fall into five camps", "five camps"),
    ("Samples fall into two camps", "two camps"),
]
OUT = Path(__file__).parent / "figure.png"


def sort_rows(p, key):
    """Order samples so camps sit together and a cloud runs from most to least A."""
    if "camps" in key:
        return p[np.lexsort((p[:, 1], p.argmax(1)))]
    return p[np.argsort(-p[:, 0])]


def draw_panel(ax, title, p):
    """Draw one ensemble as stacked bars, one row per sample, and return its measures."""
    bias, variance, error_incoherence, _, cohesion = measure(p)
    rows = np.arange(SAMPLES)[::-1]
    left = np.zeros(SAMPLES)
    for option in range(OPTIONS):
        width = p[:, option] * 100
        ax.barh(
            rows, width, left=left, color=COLOURS[option],
            height=0.74, edgecolor=SURFACE, linewidth=0.9,
        )  # fmt: skip
        left += width
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, SAMPLES - 0.2)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0%", "50%", "100%"], color=MUTED, fontsize=10)
    for label, align in zip(ax.get_xticklabels(), ["left", "center", "right"]):
        label.set_horizontalalignment(align)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=46)
    ax.text(
        0, 1.085, f"Error incoherence (Hot Mess) {error_incoherence:.2f}",
        transform=ax.transAxes, color=MUTED, fontsize=10.5,
    )  # fmt: skip
    ax.text(
        0, 1.03, f"Cohesion {cohesion:.2f}",
        transform=ax.transAxes, color=INK, fontsize=10.5,
    )  # fmt: skip
    return bias, variance


def main():
    """Draw the three ensembles side by side and save the figure."""
    z = shapes(np.random.default_rng(0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.6), sharey=True, facecolor=SURFACE)
    measures = [
        draw_panel(ax, title, sort_rows(build(z[key]), key))
        for ax, (title, key) in zip(axes, PANELS)
    ]
    bias, variance = measures[0]
    assert np.allclose(measures, measures[0]), "panels should share bias and variance"
    axes[0].set_ylabel(f"{SAMPLES} samples, one row each", color=MUTED, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS]
    fig.legend(
        handles, ["A (correct)", "B", "C", "D"],
        title="Probability the sample gives each option",
        loc="lower center", ncol=4, frameon=False,
        fontsize=11, labelcolor=INK, title_fontsize=10.5,
    )  # fmt: skip
    fig.suptitle(
        f"{SAMPLES} samples across a single question with the same bias and variance",
        x=0.035, ha="left", color=INK, fontsize=15, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.035, 0.915,
        f"Bias {bias:.2f} and variance {variance:.2f} in every panel, by the Hot Mess KL formula",
        color=MUTED, fontsize=11,
    )  # fmt: skip
    fig.subplots_adjust(left=0.05, right=0.98, top=0.77, bottom=0.15, wspace=0.1)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
