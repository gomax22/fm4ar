"""
Define methods to draw a sample from the prior.

This implementation is based on the code from Vasist et al. (2023):
https://github.com/MalAstronomy/sbi-ear
"""

import numpy as np
from scipy.stats import uniform

from fm4ar.priors.base import BasePrior
from .base import VasistDataset, VasistDatasetConfig, load_vasist_dataset