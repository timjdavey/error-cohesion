"""Toy example: Cohesion in the Hot Mess setup (arXiv:2601.23045).

One question with four options, sampled 30 times. Each sample is a probability
vector over the options, and option A is correct.

The Hot Mess measures follow the paper's KL form, and live in `measures.py` at
the repo root so the real data uses the same code. The mean prediction is the
normalised geometric mean of the samples, bias is KL(truth || mean) and variance
is the average KL(mean || sample). Error incoherence is variance / (bias + variance).

Every ensemble is built in log space as softmax(MU + scale * shape), with the
shape centred across samples. The geometric mean is then softmax(MU) whatever
the shape, so bias is identical by construction. The scale is tuned so that
variance is identical too.

Run `uv run toy/main.py` for the three ensembles in the figure, or
`uv run toy/main.py --sweep` to repeat them over 20 seeds and a range of variances.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measures import hot_mess, measure, softmax

TRUTH = 0  # option A
OPTIONS = 4
SAMPLES = 30
MU = np.log([0.45, 0.35, 0.12, 0.08])  # log of the shared mean prediction
VARIANCE = 0.8  # KL variance used in the figure


def shapes(rng):
    """Return centred directions in log space, one row per sample, for each ensemble."""
    # The camps split the samples evenly into two and five groups.
    assert SAMPLES % 10 == 0, "SAMPLES must divide by 2 and 5"

    def jitter():
        return 0.06 * rng.standard_normal((SAMPLES, OPTIONS))

    two = np.repeat([[1, -1, 0, 0], [-1, 1, 0, 0]], SAMPLES // 2, axis=0) + jitter()
    cloud = rng.standard_normal((SAMPLES, OPTIONS))
    # Four camps each favour one option and a fifth splits between A and B,
    # since five camps can't each favour a different one of four options.
    five = (
        np.repeat(
            [
                [1.2, -0.4, -0.4, -0.4],
                [-0.4, 1.2, -0.4, -0.4],
                [-0.4, -0.4, 1.2, -0.4],
                [-0.4, -0.4, -0.4, 1.2],
                [0.6, 0.6, -0.6, -0.6],
            ],
            SAMPLES // 5,
            axis=0,
        )
        + jitter()
    )
    out = {"own mix (cloud)": cloud, "two camps": two, "five camps": five}
    return {name: z - z.mean(0) for name, z in out.items()}


def build(shape, variance=VARIANCE):
    """Scale `shape` by bisection until the ensemble has the target KL variance."""
    lo, hi = 0.0, 60.0
    for _ in range(80):
        scale = (lo + hi) / 2
        if hot_mess(softmax(MU + scale * shape), TRUTH)[1] < variance:
            lo = scale
        else:
            hi = scale
    p = softmax(MU + scale * shape)
    assert np.isclose(hot_mess(p, TRUTH)[1], variance), f"variance {variance} not reached"
    return p


def figure_table():
    """Print the three ensembles shown in the figure."""
    z = shapes(np.random.default_rng(0))
    print(
        f"{'ensemble':18} {'bias':>6} {'variance':>9} {'error incoh.':>13}"
        f" {'incoherence':>12} {'cohesion':>9}"
    )
    for name in ("own mix (cloud)", "five camps", "two camps"):
        bias, variance, error_incoherence, incoherence, cohesion = measure(build(z[name]), TRUTH)
        print(
            f"{name:18} {bias:6.2f} {variance:9.2f} {error_incoherence:13.2f}"
            f" {incoherence:12.2f} {cohesion:9.2f}"
        )


def sweep(seeds=20, variances=(0.1, 0.25, 0.5, 0.8, 1.5)):
    """Print the mean of each measure over `seeds` ensembles, for each target variance."""
    for variance in variances:
        rows = {}
        for seed in range(seeds):
            for name, z in shapes(np.random.default_rng(seed)).items():
                rows.setdefault(name, []).append(measure(build(z, variance=variance), TRUTH))
        print(f"\ntarget KL variance {variance}")
        print(
            f"{'ensemble':22} {'bias':>6} {'variance':>9} {'error incoh.':>13}"
            f" {'incoherence':>12} {'cohesion':>9}"
        )
        for name, values in rows.items():
            m = np.mean(values, 0)
            print(
                f"{name:22} {m[0]:6.2f} {m[1]:9.2f} {m[2]:13.2f} {m[3]:12.2f} {m[4]:9.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="repeat over seeds and variances")
    args = parser.parse_args()
    if args.sweep:
        sweep()
    else:
        figure_table()
