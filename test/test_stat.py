"""Unit tests for intelligence.stat."""


import unittest
from intelligence import stat


class NormalDistributionTest(unittest.TestCase):
    """Test the normal distribution function."""

    def test_variance(self):
        """Test evaluation of the variance of a normal distribution."""
        normal = stat.NormalDistribution(0, 5)
        self.assertEqual(normal.variance, 25)

    def test_02(self):
        """Test evaluation of mu=0, sigma=2."""
        normal = stat.NormalDistribution(0, 2)
        self.assertEqual(normal(-1.5), 0.15056871607740221)
        self.assertEqual(normal(0), 0.19947114020071635)
        self.assertEqual(normal(10), 7.4335975736714884e-07)
