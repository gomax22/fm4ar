"""
Utility functions for sampling.
"""

import numpy as np
from scipy.special import logsumexp


def clip_and_normalize_weights(
    raw_log_weights: np.ndarray,
    percentile: float | None = None,
) -> np.ndarray:
    """
    Normalize the raw log-weights (and optionally clip them first).

    Clipping the largest raw log-weights can reduce the variance of the
    weights, at the cost of introducing a bias. Use with caution!

    Args:
        raw_log_weights: Raw log-weights.
        percentile: (Upper) percentile for clipping. If `None`, no
            clipping is applied (this is the default).

    Returns:
        normalized_weights: Normalized importance sampling weights.
    """

    # Clip the raw log-weights, if desired
    if percentile is not None:
        threshold = np.percentile(raw_log_weights, percentile)
        clipped_weights = np.clip(raw_log_weights, None, threshold)
    else:
        clipped_weights = raw_log_weights

    # Normalize the clipped log-weights
    # In "normal" space, we normalize the raw weights such that they sum to
    # the number of samples, that is:
    #   n_i = w_i * N / sum(w_i) ,
    # where N is the number of samples. We now only have access to the log-
    # weights log(w_i), so we use the following equivalent expression:
    #   n_i = exp{ log(N) + log(w_i) - LSE(log(w_i)) } ,
    # where the LSE is the log-sum-exp function:
    #   LSE(x)) = log{ sum(exp(x)) } .
    # Using the log-sum-exp trick, we can compute the LSE in a numerically
    # stable way. This allows to compute the normalized weights without
    # ever needing access to the (raw) likelihoods, priors, or proposals.
    # For more details about the log-sum-exp trick, see, e.g.,:
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    N = len(raw_log_weights)
    normalized_weights = np.exp(
        np.log(N) + clipped_weights - logsumexp(clipped_weights)
    )

    return np.array(normalized_weights)
