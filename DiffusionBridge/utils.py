"""
A module of utility functions.
"""

import torch
import math


def normal_logpdf(x, mu, sigmasq):
    """
    Evaluate log-density function of a normal distribution.

    Parameters
    ----------
    x : evaluation states (N x d)

    mu : mean vector (of size d or N x d)

    sigmasq : scalar variance

    Returns
    -------
    logdensity : log-density values(N, 1)
    """

    d = x.shape[1]
    if len(sigmasq.shape) == 0:
        constants = -0.5 * d * torch.log(
            torch.tensor(2 * math.pi)
        ) - 0.5 * d * torch.log(sigmasq)
        logdensity = constants - 0.5 * torch.sum((x - mu) ** 2, 1) / sigmasq

    if len(sigmasq.shape) == 1:
        constants = -0.5 * d * torch.log(torch.tensor(2 * math.pi)) - 0.5 * torch.sum(
            torch.log(sigmasq)
        )
        logdensity = constants - 0.5 * torch.sum((x - mu) ** 2 / sigmasq, 1)

    return logdensity


def mv_normal_logpdf(x, mu, eigvals, eigvecs):
    """
    Evaluate log-density function of a multivariate normal distribution.

    Parameters
    ----------
    x : evaluation states (N x d)

    mu : mean vector (of size d or N x d)

    eigvals : eigenvalues of covariance matrix

    eigvecs : eigenvectors of covariance matrix

    Returns
    -------
    logdensity : log-density values(N, 1)
    """

    d = len(eigvals)
    log_det_cov = torch.sum(torch.log(eigvals))
    log_constant = (
        -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi)) - 0.5 * log_det_cov
    )
    diff = x - mu
    inv_cov = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
    exponent = -0.5 * torch.sum((diff @ inv_cov) * diff, axis=1)
    logdensity = log_constant + exponent

    return logdensity
