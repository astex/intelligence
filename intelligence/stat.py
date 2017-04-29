"""Statistics utilities."""


import numpy as np


class NormalDistribution(object):
    """A normal distribution."""

    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation

    @property
    def variance(self):
        return self.standard_deviation ** 2

    def __call__(self, x):
        return (
            1. / np.sqrt(2. * np.pi * self.variance) *
            np.exp(- (x - self.mean) ** 2 / (2. * self.variance)))


GaussianDistribution = NormalDistribution
