import numpy as np
from numba import njit
from numpy import array as array_t
from collections import namedtuple


# Add documentation to the NamedTuple.
__pdoc__ = {}

# Module level declaration.
MetalAtom = namedtuple("MetalAtom",
                       ("ID", "NAME", "X", "Y", "Z", "TYPE", "DIST"))

# Add documentation for the fields.
__pdoc__["MetalAtom.X"] = "Metal atom X coordinate."
__pdoc__["MetalAtom.Y"] = "Metal atom Y coordinate."
__pdoc__["MetalAtom.Z"] = "Metal atom Z coordinate."
__pdoc__["MetalAtom.ID"] = "Metal atom ID (in the PDB file)."
__pdoc__["MetalAtom.NAME"] = "Metal atom name."
__pdoc__["MetalAtom.TYPE"] = "Metal atom type."
__pdoc__["MetalAtom.DIST"] = "Distance from the center of the protein."

# Target set of metal atoms. We fix this here internally.
METAL_TARGETS = ("MN", "FE", "CO", "NI", "CU", "ZN", "MO", "HO", "SC",
                 "MG", "PT", "PD", "TA", "CD", "AU", "HG", "OS", "TI",
                 "RH", "AS", "IR", "AG", "RU", "PB", "GD", "HF", "ZR",
                 "EU", "GA", "PA", "RE", "WB", "SM", "CR", "UN", "NB",
                 "TC", "LA", "P", "Y", "U", "V", "W")

# These are the atoms that should bond with the central atom.
BOND_METAL_TARGETS = ("N", "O", "S", "SE", "CL", "BR", "I")

# Tuple of class targets: This is immutable to avoid changes.
CLASS_TARGETS = ("LIN", "TRI", "TET", "SPL", "SQP", "TBP", "OCT")

@njit(fastmath=True)
def fast_euclidean_norm(x: array_t) -> float:
    """
    Computes the Euclidean norm using Numba fast code.
    This implementation is > 26 times faster than the
    numpy norm() version.

    :param x: is the difference vector for which we
    want to compute the norm. I.e. x = (a-b).

    :return: Euclidean norm of the input vector.
    """

    # Return the result.
    return np.sqrt(np.dot(x, x))
# _end_def_

@njit(fastmath=True)
def fast_compute_angle(pt_A: array_t, pt_B: array_t, pt_C: array_t) -> float:
    """
    Calculates the angle among 3 points in 3D (x, y, z). If the inputs are
    the points: A-B-C, it will return the angle from the vectors: BA to BC.

    :param pt_A: First point  (x1, y1, z1).

    :param pt_B: Second point (x2, y2, z2) <-- here we compute the angle.

    :param pt_C: Third point  (x3, y3, z3).

    :return: Angle (in degrees) or '0'.
    """

    # Distance from 'B' to 'A'.
    BA = pt_A - pt_B

    # Compute the Euclidean norm of 'BA'.
    norm_BA = np.sqrt(np.dot(BA, BA))

    # Distance from 'B' to 'C'.
    BC = pt_C - pt_B

    # Compute the Euclidean norm of 'BC'.
    norm_BC = np.sqrt(np.dot(BC, BC))

    # If the norms {BA, BC} are positive continue.
    if (norm_BA > 0.0) and (norm_BC > 0.0):

        # Get the cosine of the angle.
        angle_cosine = np.dot(BA, BC) / (norm_BA * norm_BC)

        # Make sure  the cosine is  in the range [-1.0, 1.0].
        # This will correct for small over/under flow errors.
        angle_cosine = np.minimum(1.0, np.maximum(-1.0, angle_cosine))

        # Get the angle in radians.
        angle_radians = np.arccos(angle_cosine)

        # Return the angle in degrees.
        return (180.0 * angle_radians) / np.pi
    else:
        # Otherwise, return 0.
        return 0.0

    # _end_if_
# _end_def_

@njit(fastmath=True)
def fast_entropy(x: array_t) -> float:
    """
    Calculate the entropy of a distribution for
    given probability values.

    :param x: input array (assuming probabilities).

    :return: the Shannon Entropy.

    NB: This is equivalent to scipy.stats.entropy.
    """

    # Make sure the probabilities are positive.
    x = np.abs(x)

    # Exclude zero entries.
    not_zero = x != 0.0

    # Compute the entropy (using the probabilities).
    y_entropy = -np.sum(x[not_zero] * np.log(x[not_zero]))

    # The Shannon Entropy value.
    return y_entropy
# _end_def_

# _end_module_
