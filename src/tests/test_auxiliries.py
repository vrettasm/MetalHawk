import unittest
import numpy as np
from scipy.stats import entropy

from src.metal_auxiliaries import (CLASS_TARGETS, fast_entropy,
                                   fast_compute_angle)


class TestAuxiliaries(unittest.TestCase):

    def test_class_targets(self):

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

        # Create a random vector.
        x = np.random.rand(10)

        # Normalize the values.
        x /= np.sum(x)

        # Check if they are equal to 7 decimal places.
        self.assertAlmostEqual(entropy(x), fast_entropy(x))

        # We set a vector with equal probability values.
        z = np.ones(10) / 10

        # In this case the entropy is maximum.
        MAX_ENTROPY_Z = float(2.3025850929940455)

        # Check if they are equal to 7 decimal places.
        self.assertAlmostEqual(MAX_ENTROPY_Z, fast_entropy(z))
    # _end_def_

    def test_angles(self):

        # This point is at the origin.
        pt0 = np.array([0.0, 0.0, 0.0])

        # This is at x=1.0 .
        pt1 = np.array([1.0, 0.0, 0.0])

        # This is at z=1.0.
        pt2 = np.array([0.0, 0.0, 1.0])

        # The angle at the origin should be exactly 90 degrees.
        self.assertEqual(90.0, fast_compute_angle(pt1, pt0, pt2))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
