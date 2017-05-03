"""Utilities for unit testing."""


import contextlib
import unittest


class TestCase(unittest.TestCase):
    """A base test case class."""

    @contextlib.contextmanager
    def assert_raises(self, exception):
        """Assert that an error is raised."""
        try:
            yield
            self.fail("No exception was raised.")
        except exception:
            pass
        except Exception as e:
            self.fail(
                "Wrong error type raised. Actual: %s, Expected: %s" % (
                    e, exception))
