"""Cohesion on the Hot Mess paper's own GPQA runs.

For each config, every question has about 40 samples, one per run, and each
sample is four probabilities. We work out the paper's KL bias, variance and
error incoherence per question, with the paper's smoothing, then Incoherence
and Cohesion from the `incoherence` package.

Run `uv run real/main.py` for the pilot config, or `uv run real/main.py --config NAME`
for another. Per-question results go to `real/results/<config>.csv`.
"""

import argparse
import csv
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load import load
from measures import PAPER_EPS, measure

RESULTS = Path(__file__).parent / "results"
PILOT = "claude-sonnet-4-gpqa-prob-fewshot-cot-1k-2"
VARIES = 1e-6  # KL variance above this means the samples disagree
BANDS = [0, 0.2, 0.4, 0.6, 0.8, 1.01]  # error incoherence bands for the report
FIELDS = [
    "doc_id", "gold", "samples", "accuracy", "bias", "variance", "error_incoherence",
    "incoherence", "cohesion", "unparsed",
]  # fmt: skip


def question_row(q):
    """Return every measure for one question as a dict."""
    bias, variance, error_incoherence, incoherence, cohesion = measure(q.probs, q.gold, PAPER_EPS)
    return {
        "doc_id": q.doc_id,
        "gold": q.gold,
        "samples": len(q.probs),
        "accuracy": (q.probs.argmax(1) == q.gold).mean(),
        "bias": bias,
        "variance": variance,
        "error_incoherence": error_incoherence,
        "incoherence": incoherence,
        "cohesion": cohesion,
        "unparsed": int((q.parse == "none").sum()),
    }


def run(config):
    """Measure every question in `config`, write the CSV and return the rows."""
    rows = [question_row(q) for q in load(config).values()]
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"{config}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    return rows


def report(rows):
    """Print how widely Cohesion spreads among questions with similar error incoherence."""
    c = {k: np.array([r[k] for r in rows], float) for k in FIELDS}
    varies = c["variance"] > VARIES
    print(f"{int(varies.sum())} of {len(rows)} questions have samples that disagree\n")
    print(f"{'error incoherence':18} {'questions':>9}  {'cohesion p10':>12} {'p50':>5} {'p90':>5}")
    for lo, hi in pairwise(BANDS):
        s = varies & (c["error_incoherence"] >= lo) & (c["error_incoherence"] < hi)
        p10, p50, p90 = np.quantile(c["cohesion"][s], [0.1, 0.5, 0.9])
        print(
            f"{lo:.1f} to {min(hi, 1):.1f}{'':8} {int(s.sum()):9}  {p10:12.2f} {p50:5.2f} {p90:5.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=PILOT, help="folder under mcq/ to run")
    args = parser.parse_args()
    print(args.config)
    report(run(args.config))
