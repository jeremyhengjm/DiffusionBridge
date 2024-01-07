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
        constants = - 0.5 * d * torch.log(torch.tensor(2 * math.pi)) - 0.5 * d * torch.log(sigmasq)
        logdensity = constants - 0.5 * torch.sum((x - mu)**2, 1) / sigmasq
        
    if len(sigmasq.shape) == 1:
        constants = - 0.5 * d * torch.log(torch.tensor(2 * math.pi)) - 0.5 * torch.sum(torch.log(sigmasq))
        logdensity = constants - 0.5 * torch.sum((x - mu)**2 / sigmasq, 1) 
    
    return logdensity

    
