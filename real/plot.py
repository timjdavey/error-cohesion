"""Draw three real GPQA questions from Sonnet 4's runs at a 1k thinking budget.

Run `uv run real/plot.py` to write `real/figure.png`, drawn like the toy.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load import load
from main import PILOT, question_row

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # same as toy/plot.py
OUT = Path(__file__).parent / "figure.png"

#: Three questions with error incoherence between 0.5 and 0.57, picked from the
#: 0.4 to 0.6 band as the lowest Cohesion, a middle one and one near the top.
EXAMPLES = [
    (81, "Four loose camps, mostly unsure"),
    (407, "Two camps, mostly certain"),
    (118, "Two camps, always certain"),
]


def sort_rows(p):
    """Group samples by their top answer, then by how sure they are of it."""
    return p[np.lexsort((-p.max(1), p.argmax(1)))]


def style(ax):
    """Strip an axis back to the toy's look."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(length=0, colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_question(ax, q, title):
    """Draw one question's samples as stacked bars, one row per run."""
    row = question_row(q)
    p = sort_rows(q.probs)
    n = len(p)
    rows, left = np.arange(n)[::-1], np.zeros(n)
    for option in range(4):
        width = p[:, option] * 100
        ax.barh(
            rows, width, left=left, color=COLOURS[option],
            height=0.74, edgecolor=SURFACE, linewidth=0.9,
        )  # fmt: skip
        left += width
    style(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0%", "50%", "100%"])
    for label, align in zip(ax.get_xticklabels(), ["left", "center", "right"]):
        label.set_horizontalalignment(align)
    ax.set_yticks([])
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=62)
    lines = [
        (f"Question {q.doc_id}, correct answer {'ABCD'[q.gold]}", MUTED),
        (f"Error incoherence (Hot Mess) {row['error_incoherence']:.2f}", MUTED),
        (f"Cohesion {row['cohesion']:.2f}", INK),
    ]
    for i, (text, colour) in enumerate(lines):
        ax.text(0, 1.14 - 0.055 * i, text, transform=ax.transAxes, color=colour, fontsize=10.5)


def figure(questions):
    """Draw the three example questions side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 7.4), sharey=True, facecolor=SURFACE)
    for ax, (doc_id, title) in zip(axes, EXAMPLES):
        draw_question(ax, questions[doc_id], title)
    axes[0].set_ylabel("40 runs, one row each", color=MUTED, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS]
    fig.legend(
        handles, ["A", "B", "C", "D"],
        title="Probability the run gives each option",
        loc="lower center", ncol=4, frameon=False,
        fontsize=11, labelcolor=INK, title_fontsize=10.5,
    )  # fmt: skip
    fig.suptitle(
        "Three real GPQA questions with about the same error incoherence",
        x=0.035, ha="left", color=INK, fontsize=15, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.035, 0.925, "Claude Sonnet 4 with a 1k thinking budget, from the Hot Mess paper's own runs",
        color=MUTED, fontsize=11,
    )  # fmt: skip
    fig.subplots_adjust(left=0.05, right=0.98, top=0.76, bottom=0.14, wspace=0.1)
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)


def main():
    """Draw the three example questions."""
    figure(load(PILOT))
    print(OUT)


if __name__ == "__main__":
    main()
