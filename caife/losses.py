"""Module contaning common loss functions."""

import numpy as np
from numpy.typing import ndarray

def poisson_nll(model_output: ndarray, proxy_output: ndarray, sample_weight: ndarray | None = None):
    """
    g_est     <- model_output
    g_truth   <- proxy_output?

    P(g_truth | g_est) = g_est^g_truth * exp(-g_est) / g_truth!    <- likelihood function Poisson dist.

    log(P) = g_truth * log(g_est) - g_est - log(g_truth!)          <- apply log
 
    log(P) = g_truth * log(g_est) - g_est                          <- remove constant term since optimization stays the same

    -log(P) = g_est - g_truth * log(g_est)                         <- multiply with -1
    
    """
    eps = 1e-8
    
    loss = model_output - proxy_output * np.log(model_output + eps)
    
    # TODO evaluate whether to use np.average or np.sum
    """
    np.average normalizes the weights to sum up to 1.
    => if the provided sample weights are already normalized
    this solution should be equivalent to:
        >>loss *= sample_weight
        >>return np.sum(loss)
    otherwise it is equivalent to:
        >>loss *= sample_weight / sample_weight.sum()
        >>return np.sum(loss)
    """
    return np.average(loss, weights=sample_weight)

