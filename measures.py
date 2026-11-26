"""The measures shared by the toy and the real data.

The Hot Mess measures follow the paper's KL form (arXiv:2601.23045). The mean
prediction is the normalised geometric mean of the samples, bias is
KL(truth || mean) and variance is the average KL(mean || sample). Error
incoherence is variance / (bias + variance).

Incoherence and Cohesion come from the `incoherence` package, with each sample
treated as one trial.
"""

import incoherence as ic
import numpy as np

#: Laplace smoothing the paper uses before taking logs (EPS in its scripts/utils.py).
PAPER_EPS = 1e-12


def softmax(x):
    """Turn rows of log weights into probability vectors."""
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def smooth(p, eps):
    """Add `eps` to every option and renormalise, as the paper does, so logs stay finite."""
    return (p + eps) / (1 + p.shape[-1] * eps)


def hot_mess(p, truth=0, eps=0.0):
    """Return bias, variance and error incoherence for samples `p` (samples x options).

    `truth` is the index of the correct option. Set `eps` to smooth zeros first,
    which real answers need and the toy does not.
    """
    if eps:
        p = smooth(p, eps)
    mean = softmax(np.log(p).mean(0))
    bias = -np.log(mean[truth])
    variance = max(0.0, (mean * (np.log(mean) - np.log(p))).sum(1).mean())
    error = bias + variance
    return bias, variance, variance / error if error > 0 else 0.0


def measure(p, truth=0, eps=0.0):
    """Return the Hot Mess measures, then Incoherence and Cohesion, for samples `p`."""
    result = ic.measure_counts(p)  # each sample is one trial
    return (*hot_mess(p, truth, eps), result.incoherence, result.cohesion)
