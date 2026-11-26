"""Toy example: Cohesion in the Hot Mess setup (arXiv:2601.23045).

One question with four options, sampled 30 times. Each sample is a probability
vector over the options, and option A is correct.

The Hot Mess measures follow the paper's KL form. The mean prediction is the
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

import incoherence as ic
import numpy as np

TRUTH = 0  # option A
OPTIONS = 4
SAMPLES = 30
MU = np.log([0.45, 0.35, 0.12, 0.08])  # log of the shared mean prediction
VARIANCE = 0.8  # KL variance used in the figure


def softmax(x):
    """Turn rows of log weights into probability vectors."""
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def hot_mess(p):
    """Return bias, variance and error incoherence for samples `p` (samples x options)."""
    mean = softmax(np.log(p).mean(0))
    bias = -np.log(mean[TRUTH])
    variance = (mean * (np.log(mean) - np.log(p))).sum(1).mean()
    return bias, variance, variance / (bias + variance)


def shapes(rng):
    """Return centred directions in log space, one row per sample, for each ensemble."""

    def jitter():
        return 0.06 * rng.standard_normal((SAMPLES, OPTIONS))

    two = np.repeat([[1, -1, 0, 0], [-1, 1, 0, 0]], SAMPLES // 2, axis=0) + jitter()
    three = (
        np.repeat(
            [[1.2, -0.6, -0.6, 0], [-0.6, 1.2, -0.6, 0], [-0.6, -0.6, 1.2, 0]],
            SAMPLES // 3,
            axis=0,
        )
        + jitter()
    )
    cloud = rng.standard_normal((SAMPLES, OPTIONS))
    line = np.outer(np.linspace(-1, 1, SAMPLES), [1, -1, 0, 0]) + jitter()
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
    out = {
        "own mix (cloud)": cloud,
        "own mix (A to B line)": line,
        "two camps": two,
        "three camps": three,
        "five camps": five,
    }
    return {name: z - z.mean(0) for name, z in out.items()}


def build(shape, mu=MU, variance=VARIANCE):
    """Scale `shape` by bisection until the ensemble has the target KL variance."""
    lo, hi = 0.0, 60.0
    for _ in range(80):
        scale = (lo + hi) / 2
        if hot_mess(softmax(mu + scale * shape))[1] < variance:
            lo = scale
        else:
            hi = scale
    return softmax(mu + scale * shape)


def measure(p):
    """Return the Hot Mess measures, then Incoherence and Cohesion, for samples `p`."""
    result = ic.measure_counts(p)  # each sample is one trial
    return (*hot_mess(p), result.incoherence, result.cohesion)


def figure_table():
    """Print the three ensembles shown in the figure."""
    z = shapes(np.random.default_rng(0))
    print(
        f"{'ensemble':18} {'bias':>6} {'variance':>9} {'error incoh.':>13}"
        f" {'incoherence':>12} {'cohesion':>9}"
    )
    for name in ("own mix (cloud)", "five camps", "two camps"):
        bias, variance, error_incoherence, incoherence, cohesion = measure(build(z[name]))
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
                p = build(z, variance=variance)
                certain = (p.max(1) > 0.95).mean()  # share of samples near certain
                rows.setdefault(name, []).append([*measure(p), certain])
        print(f"\ntarget KL variance {variance}")
        print(
            f"{'ensemble':22} {'bias':>6} {'variance':>9} {'error incoh.':>13}"
            f" {'incoherence':>12} {'cohesion':>14} {'near certain':>13}"
        )
        for name, values in rows.items():
            m, sd = np.mean(values, 0), np.std(values, 0)
            print(
                f"{name:22} {m[0]:6.2f} {m[1]:9.2f} {m[2]:13.2f}"
                f" {m[3]:12.2f} {m[4]:8.2f} ±{sd[4]:.2f} {m[5]:13.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="repeat over seeds and variances")
    args = parser.parse_args()
    sweep() if args.sweep else figure_table()
