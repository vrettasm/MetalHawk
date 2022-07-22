"""
    Test the methods in the metal_auxiliaries module.
"""

import unittest
import numpy as np
from scipy.stats import entropy

from src.metal_auxiliaries import (CLASS_TARGETS, fast_entropy,
                                   fast_compute_angle)


class TestAuxiliaries(unittest.TestCase):

    def test_class_targets(self):
        """
        Test the CLASS_TARGET list. It should be in a
        very specific order otherwise the predictions
        will be wrong.

        :return: None.
        """

        # The order of the CLASS_TARGETS should be identical with:
        _test_targets = ("LIN", "TRI", "TET", "SPL", "SQP", "TBP",
                         "OCT")
        # Define a flag.
        are_equal = True

        # Initialize with None.
        error_message = None

        # Check them one by one.
        for target_1, target_2 in zip(CLASS_TARGETS, _test_targets):

            # Check for inequality.
            if target_1 != target_2:

                # Change the flag value.
                are_equal = False

                # Construct the error message.
                error_message = f"{target_1} not equal to {target_2}"

                # Skip the rest of the targets.
                break
            # _end_if_

        # _end_if_

        # Assertion if they are not the identical.
        self.assertTrue(are_equal, msg=error_message)
    # _end_def_

    def test_entropy(self):
        """
        Test the fast_entropy method. Here we test:

            1) that the method works correctly compared
            to scipy methods

            2) that the method returns the correct max
            value for a test case where we can compute
            its entropy analytically

        :return: None.
        """

        # Create a random vector from U(0,1).
        x = np.random.rand(10)

        # Normalize the values to sum to one.
        x /= np.sum(x)

        # Perturb the entropy by altering the
        # probability values at two entries in
        # the array.
        x[0] += x[1]
        x[0] += x[4]

        # We must set these entries to zero
        # to still account for probability.
        x[1] = 0.0
        x[4] = 0.0

        # Check if they are equal to 7 decimal places.
        self.assertAlmostEqual(entropy(x), fast_entropy(x))

        # We set a vector with equal probability values.
        z = np.ones(10) / 10

        # In this scenario the entropy is maximum.
        MAX_ENTROPY_Z = float(2.3025850929940455)

        # Check if they are equal to 7 decimal places.
        self.assertAlmostEqual(MAX_ENTROPY_Z, fast_entropy(z))
    # _end_def_

    def test_angles(self):
        """
        Test the angles' calculation.

        :return: None.
        """

        # This point is at the origin.
        pt0 = np.array([0.0, 0.0, 0.0])

        # This is at x=1.0.
        pt1 = np.array([1.0, 0.0, 0.0])

        # This is at z=1.0.
        pt2 = np.array([0.0, 0.0, 1.0])

        # The angle at the origin should be exactly 90 degrees.
        self.assertEqual(90.0, fast_compute_angle(pt1, pt0, pt2))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
