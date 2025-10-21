"""
Tests for `fm4py.utils.sampling`.
"""

import numpy as np

from fm4ar.utils.sampling import clip_and_normalize_weights


def test__clip_and_normalize_weights() -> None:
    """
    Test `clip_and_normalize_weights()`.
    """

    rng = np.random.default_rng(42)
    raw_log_weights = np.log10(rng.uniform(0, 1, 10_000))

    # Case 1
    normalized_weights = clip_and_normalize_weights(
        raw_log_weights=raw_log_weights,
        percentile=None,
    )
    assert np.allclose(np.max(normalized_weights), 1.4382241623505292)

    # Case 2
    normalized_weights = clip_and_normalize_weights(
        raw_log_weights=raw_log_weights,
        percentile=0.95,
    )
    assert np.allclose(np.max(normalized_weights), 1.0028639021473102)