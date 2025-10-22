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

    if sample_weight is not None:
        loss *= sample_weight 
    
    return np.sum(loss)    # <- funfolding does this, does np.mean make sense here?

