"""Incoherence next to the Hot Mess measures (arXiv:2601.23045), on toy ensembles.

Both read a set of samples, where each sample is a probability vector over the
options of one question. The code calls a set an ensemble. They answer different
questions.

- Incoherence (Davey, Entropy 2024) is how much the samples disagree with each
  other. It never uses the correct answer.
- Error incoherence (Hot Mess) is the share of the error that is variance. It is
  variance / (bias + variance), so it needs the correct answer.

The Hot Mess paper gives two forms of variance. The KL form uses the normalised
geometric mean of the samples and averages KL(mean || sample). The Brier form
uses the plain mean and averages the squared distance to it.

Run `uv run compare/main.py` to print every table in the README.
"""

import incoherence as ic
import numpy as np

TRUTH = 0  # option A
SAMPLES = 30
DELTA = 1e-6  # smoothing for the KL form, so a certain sample has no log(0)
COUNTS = (10, 100, 1000, 10000)  # ensemble sizes for the odd sample tables
MILLION = 1_000_000  # one more size, to show Incoherence still falls slowly


def incoherence(p):
    """Return Incoherence for samples `p` (samples x options)."""
    return ic.measure_counts(p).incoherence  # each sample is one trial


def entropies(p):
    """Return the entropy of each sample and the entropy of their pool, in bits.

    The pool is the plain mean of the samples. Incoherence is built from the gap
    between these two.
    """
    result = ic.measure_counts(p)
    return result.entropies, result.pooled_entropy


def brier_variance(p, scaled=True):
    """Return the mean squared distance from each sample to the mean sample.

    The largest value is 1 - 1/options, which certain samples spread evenly over
    every option reach. `scaled` divides by that, so the result runs from 0 to 1.
    """
    p = np.asarray(p, dtype=float)
    variance = ((p - p.mean(0)) ** 2).sum(1).mean()
    return variance / (1 - 1 / p.shape[1]) if scaled else variance


def kl_parts(p, delta=DELTA):
    """Return the Hot Mess KL bias and variance for samples `p`.

    `delta` is added to every probability before the samples are scaled back to
    sum to 1. Without it, the geometric mean is zero wherever one sample is zero.
    """
    p = np.asarray(p, dtype=float) + delta
    logs = np.log(p / p.sum(1, keepdims=True))
    mean = np.exp(logs.mean(0))
    mean /= mean.sum()
    bias = -np.log(mean[TRUTH])
    variance = (mean * (np.log(mean) - logs)).sum(1).mean()
    return bias, variance


def error_incoherence(p):
    """Return the share of the KL error that is variance."""
    bias, variance = kl_parts(p)
    return variance / (bias + variance)


def certain(option, options=4):
    """Return a sample that puts everything on one option."""
    return np.eye(options)[option]


def confident(option, options=4, p=0.97):
    """Return a sample that puts `p` on one option and shares the rest evenly."""
    out = np.full(options, (1 - p) / (options - 1))
    out[option] = p
    return out


def wobble(option):
    """Return samples that all favour one option, each nudged a little in log space.

    The nudges are the same whichever option is favoured, so two calls with
    different options disagree among themselves by exactly the same amount.
    """
    noise = np.random.default_rng(0).standard_normal((SAMPLES, 4))
    logits = np.log(confident(0)) + noise - noise.mean(0)
    e = np.exp(logits - logits.max(1, keepdims=True))
    order = np.arange(4)
    order[[0, option]] = option, 0  # swap the favoured option into place
    return (e / e.sum(1, keepdims=True))[:, order]


def shifted_pair(base, shift=0.08):
    """Return two samples over two options that differ by `shift`."""
    return np.array([[base, 1 - base], [base - shift, 1 - base + shift]])


def one_odd_run(count, odd, usual):
    """Return `count` samples where all but the last are `usual` and the last is `odd`."""
    return np.vstack([np.tile(usual, (count - 1, 1)), odd])


def percents(sample):
    """Return a sample as text, like "A 97%, B 1%, C 1%, D 1%"."""
    return ", ".join(f"{name} {share:.0%}" for name, share in zip("ABCD", sample))


def pool_table():
    """Print how Incoherence is built from entropies, on two sets of two samples."""
    singles = {"certain": certain(0), "97% confident": confident(0), "even": np.full(4, 0.25)}
    print("Entropy of one sample over four options, in bits")
    for name, sample in singles.items():
        # A set of two copies of one sample pools into that sample.
        print(f"{name:14} {entropies(np.vstack([sample, sample]))[1]:6.3f}")
    sets = {
        "both 97% on A": np.vstack([confident(0), confident(0)]),
        "one 97% on A, one 97% on B": np.vstack([confident(0), confident(1)]),
    }
    print("\nHow Incoherence reads a set of two samples")
    print(f"{'set':28} {'entropy of each':>16} {'pool entropy':>13} {'gap':>6} {'incoherence':>12}")
    for name, p in sets.items():
        each, pool = entropies(p)
        print(
            f"{name:28} {each[0]:16.2f} {pool:13.2f} {pool - each[0]:6.2f} {incoherence(p):12.2f}"
        )
        print(f"{'':28} pool is {percents(p.mean(0))}")


def readings_table():
    """Print five ensembles where the two measures read the same samples differently."""
    half = SAMPLES // 2
    ensembles = {
        "always right": np.tile(confident(0), (SAMPLES, 1)),
        "always wrong": np.tile(confident(1), (SAMPLES, 1)),
        "right, with a wobble": wobble(0),
        "wrong, same wobble": wobble(1),
        "two camps, A and B": np.vstack(
            [np.tile(confident(0), (half, 1)), np.tile(confident(1), (half, 1))]
        ),
    }
    print("\nSame set, two readings (30 samples, four options, A is correct)")
    print(f"{'set':22} {'bias':>6} {'KL variance':>12} {'error incoh.':>13} {'incoherence':>12}")
    for name, p in ensembles.items():
        bias, variance = kl_parts(p)
        print(
            f"{name:22} {bias:6.2f} {variance:12.3f} {error_incoherence(p):13.2f}"
            f" {incoherence(p):12.2f}"
        )
    top = wobble(0)[:, 0]
    print(f"\nThe wobble: each sample puts {top.min():.0%} to {top.max():.0%} on its top option")
    for sample in wobble(0)[:2]:
        print(f"  {percents(sample)}")


def order_table():
    """Print two pairs of samples with the same wobble, one nearly certain and one nearly even."""
    print("\nThe same wobble of 8 percentage points between two samples, over two options")
    print(
        f"{'pair, by share on A':30} {'entropies':>13} {'pool':>6} {'incoherence':>12}"
        f" {'KL variance':>12} {'Brier':>7}"
    )
    for name, base in (("nearly certain", 0.98), ("nearly even", 0.54)):
        p = shifted_pair(base)
        each, pool = entropies(p)
        print(
            f"{f'{name}: {p[0, 0]:.0%} and {p[1, 0]:.0%}':30} {each[0]:6.3f} {each[1]:6.3f}"
            f" {pool:6.3f} {incoherence(p):12.3f} {kl_parts(p)[1]:12.3f} {brier_variance(p):7.4f}"
        )


def rare_run_table():
    """Print how each measure falls as one odd sample is diluted by more usual ones."""
    cases = {
        "One even sample, the rest certain of A": (np.full(4, 0.25), certain(0)),
        "One sample certain of B, the rest certain of A": (certain(1), certain(0)),
        "One sample 97% on B, the rest 97% on A": (confident(1), confident(0)),
    }
    for number, (title, (odd, usual)) in enumerate(cases.items()):
        print(f"\n{title}")
        print(
            f"{'samples':>10} {'pool entropy':>13} {'incoherence':>12} {'Brier':>10}"
            f" {'root Brier':>11}"
        )
        for count in COUNTS if number else (*COUNTS, MILLION):
            p = one_odd_run(count, odd, usual)
            brier = brier_variance(p)
            print(
                f"{count:10,} {entropies(p)[1]:13.4f} {incoherence(p):12.3f} {brier:10.6f}"
                f" {np.sqrt(brier):11.3f}"
            )


def certain_table():
    """Print ensembles of certain samples, where the KL form depends on the smoothing."""
    ensembles = {
        "2 options, one sample on each": np.eye(2),
        "4 options, one sample on each": np.eye(4),
        "64 options, one sample on each": np.eye(64),
        "4 options, two samples on two": np.eye(4)[:2],
        "4 options, three on A and one on B": np.eye(4)[[0, 0, 0, 1]],
    }
    print("\nCertain samples")
    print(
        f"{'set':36} {'incoherence':>12} {'Brier unscaled':>15} {'Brier':>7}"
        f" {'KL 1e-6':>8} {'KL 1e-12':>9}"
    )
    for name, p in ensembles.items():
        print(
            f"{name:36} {incoherence(p):12.3f} {brier_variance(p, scaled=False):15.3f}"
            f" {brier_variance(p):7.3f} {kl_parts(p, 1e-6)[1]:8.1f} {kl_parts(p, 1e-12)[1]:9.1f}"
        )


if __name__ == "__main__":
    pool_table()
    readings_table()
    order_table()
    rare_run_table()
    certain_table()
