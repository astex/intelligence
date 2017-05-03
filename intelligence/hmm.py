"""Tools for working with hidden markov models."""


import numpy as np


class Viterbi(object):
    """A wrapper class for running the viterbi algorithm."""
    
    def __init__(self, states, priors, transitions, emissions):
        """Initialize the viterbi wrapper.

        Args:
            states: The state space. K,
            priors: The prior probability of observing each state. K,
            transitions: The probability of transitioning from state i
                to state j. K,K
            emissions: An array of functions that returns the probability of
                observing a state given an observation. K,
        """
        self.n_states = np.shape(states)[0] # K

        if np.shape(priors) != (self.n_states,):
            raise ValueError("priors must be %s long" % self.n_states)
        if np.shape(transitions) != (self.n_states, self.n_states):
            raise ValueError(
                "transitions must be %s by %s" % (self.n_states, self.n_states))

        self.states = states
        self.priors = priors
        self.transitions = transitions
        self.emissions = emissions

    @classmethod
    def from_discrete_observations(
            cls, states, priors, transitions, observations, emissions):
        """Initialize a viterbi wrapper from a discrete set of observations.

        Args:
            states, priors, transitions: As defined in __init__.
            observations: The set of possible observations. N,
            emissions: An array of the probability of observing each state for
                each possible observation. K,N

        Returns:
            An initialized Viterbi instance.
        """
        observations = list(observations)

        def _get_emission_function(state_emissions):
            def emission_function(evidence):
                observation_index = observations.index(evidence)
                return state_emissions[observation_index]
            return emission_function

        emission_functions = [_get_emission_function(e) for e in emissions]
        return cls(states, priors, transitions, emission_functions)

    def build_probs(self, evidence):
        """Build viterbi tables from evidence.

        Args:
            evidence: A list of observations. T,

        Returns:
            (
                The probability of traversing each path. K,T
                The previous state in each path. K,T)
        """
        n_ev = np.shape(evidence)[0] # T

        probs = np.zeros((self.n_states, n_ev), dtype=np.float64)
        paths = np.zeros((self.n_states, n_ev), dtype=int)

        probs[:,0] = (
            self.priors.T *
            np.array([e(evidence[0]) for e in self.emissions]))

        for t in range(1, n_ev):
            for s in range(self.n_states):
                # Probability of transitioning to state s given previous
                # transitions. K,
                p = probs[:, t - 1] * self.transitions[:, s]
                paths[s, t] = np.argmax(p)
                probs[s, t] = p[paths[s, t]] * self.emissions[s](evidence[t])

        return probs, paths

    def build_path(self, paths, end_state):
        """Build the path traversed to the end state.

        Args:
            probs, paths: The return values of build_probs.

        Returns:
            A list of states for each evidence.
        """
        path = [end_state]
        for prev_paths in paths[:,-1:0:-1].T:
            path.insert(0, prev_paths[path[0]])
        return [self.states[s] for s in path]

    def run(self, evidence):
        """Get the best path and probability of traversing it."""
        probs, paths = self.build_probs(evidence)
        end_state = np.argmax(probs[:, -1])
        path = self.build_path(paths, end_state)
        return path, probs[end_state, -1]
