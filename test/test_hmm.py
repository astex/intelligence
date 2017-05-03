"""Unit tests for intelligence.hmm."""


from intelligence import hmm
from test import lib as test_lib
import numpy as np
import unittest


class ViterbiTest(test_lib.TestCase):
    """Unit tests for hmm.Viterbi."""

    def test_wrong_priors_len(self):
        """Test that it raises an error if priors is the wrong length."""
        with self.assertRaises(ValueError):
            hmm.Viterbi([1], [], [[1]], [lambda x: 1])

    def test_wrong_transitions_dim(self):
        """Test that it raises an error if transitions is the wrong size."""
        with self.assertRaises(ValueError):
            hmm.Viterbi([1], [1], [[]], [lambda x: 1])

    def test_onoff(self):
        """Test with a simple alternating switch."""
        states = np.array([0, 1]) # Off and on.
        priors = np.array([1, 0]) # Starts in the off position.
        transitions = np.array([[0, 1], [1, 0]]) # On > Off, Off > On
        observations = np.array([0, 1]) # Off and on.
        emissions = np.array([[1, 0], [0, 1]]) # Off emits off, on emits on.

        viterbi = hmm.Viterbi.from_discrete_observations(
            states, priors, transitions, observations, emissions)

        path, prob = viterbi.run([0, 1, 0, 1])
        self.assertEqual(path, [0, 1, 0, 1])
        self.assertEqual(prob, 1)

        path, prob = viterbi.run([1, 1])
        del path
        self.assertEqual(prob, 0)
