"""
This module includes the main class that handles the data generation from the Metal PDB files.


                                    PDB/CSD Record Format

        COLUMNS        DATA  TYPE    FIELD        DEFINITION

        -------------------------------------------------------------------------------------

         1 -  6        Record name   "ATOM  "

         7 - 11        Integer       serial       Atom serial number.

        13 - 16        Atom          name         Atom name.

        17             Character     altLoc       Alternate location indicator.

        18 - 20        Residue name  resName      Residue name.

        22             Character     chainID      Chain identifier.

        23 - 26        Integer       resSeq       Residue sequence number.

        27             AChar         iCode        Code for insertion of residues.

        31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.

        39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.

        47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.

        55 - 60        Real(6.2)     occupancy    Occupancy.

        61 - 66        Real(6.2)     tempFactor   Temperature factor.

        77 - 78        LString(2)    element      Element symbol, right-justified.

        79 - 80        LString(2)    charge       Charge on the atom.

        -------------------------------------------------------------------------------------

        NOTE: The numbering of the columns starts from '1', not '0'.

"""

# Python imports.
from collections import defaultdict
from operator import itemgetter
from os import linesep
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.spatial.qhull import QhullError

# Custom code imports.
from src.metal_auxiliaries import (MetalAtom,
                                   METAL_TARGETS,
                                   fast_compute_angle,
                                   fast_euclidean_norm)


class MetalPdbData(object):
    """
    Provides assistance with processing the metal PDB files. It generates contact/
    distance maps from the "n-closest" atoms near the center metal atom.

    Optionally, it can use the selected atoms and create a ConvexHull object, were
    we can later extract its volume and its surface area.

    Optionally, we can compute the angles among the center metal atom and its nearest
    points (excluding Hydrogen and Carbon).
    """

    # Object variables.
    __slots__ = ("_compute_angles", "_compute_convex_hull", "_dataset", "_distance_map",
                 "_volume", "_area", "_center_metal_atom", "_angles", "_stats_counter")

    def __init__(self, compute_angles=False, compute_convex_hull=False):
        """
        Constructs an object that will create the contact map, along with
        the rest of the data arrays from a given PDB / CSD (metal) file.
        Initially all the class fields are set to None.

        :param compute_angles: Boolean flag that enables the computation of
        the angles among the estimated contact map points. Default is False.

        :param compute_convex_hull: Boolean flag that enables the computation of
        a "convex hull", from the contact map points. This will allow estimating
        quantities such as the area and the volume of the points.
        Default is False.
        """

        self._compute_angles = compute_angles
        """
        This flag allows the computation of the angles
        between the selected points of the contact map.
        """

        self._compute_convex_hull = compute_convex_hull
        """
        This flag allows the computation of the convex hull
        related quantities, such as area and volume of the
        contact map points.
        """

        self._area = None
        """
        This is the area of the convex-hull object created by
        the (x,y,z) coordinates of the selected atoms, around
        the central metal atom.
        """

        self._volume = None
        """
        This is the volume of the convex-hull object, created by
        the (x,y,z) coordinates of the selected atoms around the
        central metal atom.
        """

        self._dataset = None
        """
        This is the dataset that will hold all the atoms around
        the central metal atom, along with the distances of the
        of these atoms (excluding 'H' and 'C'), from the center
        atom.
        """

        self._distance_map = None
        """
        This is the distance/contact map that has been generated
        by the PDB file using all the pairwise distances from all
        the selected metal atoms.
        """

        self._angles = None
        """
        This is a numpy array that contains the computed angles
        between all the map points.  The first three columns of
        the array contain the index positions of the points and
        the last column contains the angle in degrees.
        """

        self._center_metal_atom = None
        """
        The center metal atom is the point of reference for all
        the distance / contact maps. It is of 'MetalAtom' type.
        """

        self._stats_counter = {"n_calls": 0,
                               "metals_freq": defaultdict(int)}
        """
        This dictionary will hold statistics for the object. It
        will hold information like: 1) the number of times that
        was called, 2) the frequency of the metal site atoms it
        found, etc.
        """
    # _end_def_

    def reset_all_data(self, show_warning=False):
        """
        Clears all the data and resets them back to empty lists.
        This is to ensure that if we accidentally call the main
        function twice, we will not append the same data in the
        lists.

        :param show_warning: Boolean flag to prompt a warning
        before resetting the data. There is no option to stop
        the process.

        :return: None.
        """

        # Reset everything to None/False.
        for field in self.__slots__:

            # Skip the flags and the statistics dictionary.
            if field in {"_stats_counter", "_compute_angles",
                         "_compute_convex_hull"}:
                continue
            # _end_if_

            # Make sure the field exists.
            if hasattr(self, field):

                # Reset its value to None.
                setattr(self, field, None)

            # _end_if_

        # _end_for_

        # Check if we want to display a warning.
        if show_warning:
            print(f"WARNING: {self.__class__.__name__}: "
                  f"All the data have been reset to 'None'!")
        # _end_if_

    # _end_def_

    @property
    def compute_angles(self):
        """
        Accessor (getter) of the compute_angles flag.

        :return: the boolean value of the compute_angles flag.
        """
        return self._compute_angles

    # _end_def_

    @compute_angles.setter
    def compute_angles(self, new_value):
        """
        Accessor (setter) of the compute_angles flag.

        :param new_value: (bool).
        """

        # Check for correct type.
        if isinstance(new_value, bool):

            # Update the flag value.
            self._compute_angles = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Compute angles flag should be bool: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def compute_convex_hull(self):
        """
        Accessor (getter) of the compute_convex_hull flag.

        :return: the boolean value of the compute_convex_hull flag.
        """
        return self._compute_convex_hull

    # _end_def_

    @compute_convex_hull.setter
    def compute_convex_hull(self, new_value):
        """
        Accessor (setter) of the compute_convex_hull flag.

        :param new_value: (bool).
        """

        # Check for correct type.
        if isinstance(new_value, bool):

            # Update the flag value.
            self._compute_convex_hull = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Compute convex hull flag should be bool: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def get_statistics(self):
        """
        Provides access to the statistics that have
        been collected during the current life-time
        of the object.

        :return: the dictionary with the statistics.
        """
        return self._stats_counter
    # _end_def_

    @property
    def dataset(self):
        """
        Accessor (getter) of the dataset.

        :return:  a tuple with a pandas DataFrame containing ALL
        the atoms along with a list with the sorted distances of
        the non hydrogen atoms from the central metal atom.
        """

        # Check if the dataset exists.
        if self._dataset is not None:

            # Return all the data.
            return self._dataset
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Dataset is not initialized yet.")
        # _end_if_

    # _end_def_

    @property
    def distance_map(self):
        """
        Accessor (getter) of the distance/contact map.

        :return: a symmetric [N x N] map with zeros in
        the diagonal.
        """

        # Check if the distance map exist.
        if self._distance_map is not None:

            # Return the map.
            return self._distance_map
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Distance map has not been estimated.")
        # _end_if_

    # _end_def_

    @property
    def min_length(self):
        """
        Accessor (getter) of the number of contacts that
        have been estimated. I.e. the number of atoms we
        have compared the central atom with.

        :return: the height (or the width) of the distance
        map.
        """

        # Check if the distance map exist.
        if self._distance_map is not None:

            # Return the height of the map.
            return self._distance_map.shape[0]
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Distance map has not been estimated.")
        # _end_if_

    # _end_def_

    @property
    def angles(self):
        """
        Accessor (getter) of the calculated angles.

        :return: an array of angles between the contact
        map points.
        """

        # Check if the angles list exist.
        if self._angles is not None:

            # Return the list.
            return self._angles
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Angles array has not been estimated.")
        # _end_if_

    # _end_def_

    @property
    def convex_area(self):
        """
        Accessor (getter) of the convex-hull surface area from
        all the selected points around the central metal atom.

        :return: a list of floats in $L^2$ units [L: Angstrom].
        """

        # Check if the area list is empty.
        if self._area is not None:

            # Return the list with the surface areas.
            return self._area
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Convex-hull surface areas "
                               f"have not been estimated.")
        # _end_if_

    # _end_def_

    @property
    def convex_volume(self):
        """
        Accessor (getter) of the convex-hull volume from
        all the selected atoms around the central metal.

        :return: a list of floats in $L^3$ units.

        --> L: should be in Angstrom.
        """

        # Check if the volume list is empty.
        if self._volume is not None:

            # Return the convex volume.
            return self._volume
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Convex-hull volumes list "
                               f"has not been estimated.")
        # _end_if_

    # _end_def_

    @property
    def center_metal_atom(self):
        """
        Accessor (getter) of the center metal atom.

        :return: a MetalAtom (namedtuple).
        """

        # Check if atom exists.
        if self._center_metal_atom is not None:

            # Return the MetalAtom object.
            return self._center_metal_atom
        else:

            # Raise an error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Central metal atom is not found yet.")
        # _end_if_

    # _end_def_

    @staticmethod
    def angles_from_coordinates(points_xyz, n_closest=None):
        """
        Computes a list of angles among the points in the input array.
        We always assume that the first item in the "points_xyz" has
        the central atom.

        :param points_xyz: Input array with the (x, y, z) coordinates.
        Each row in the input array is a different point (in 3D space).

        :param n_closest: Number of neighbouring points to consider when
        computing the angles. If 'None', then we will compute and return
        all the angles.

        :return: A list with the angles.
        """

        # Total number of points in the input list.
        L = len(points_xyz)

        # If there are less than three atoms we can't
        # compute any angles.
        if L < 3:
            return None
        # _end_if_

        # Check if an input is given.
        if n_closest is None:
            # Maximum number of points to consider.
            n_closest = L-1
        else:
            # Make sure we are not out of bounds.
            n_closest = min(L-1, max(1, n_closest))
        # _end_if_

        # Ensure the input is int.
        n_closest = int(n_closest)

        # Store the angles.
        angles_list = []

        # Localize the list append method.
        angles_list_append = angles_list.append

        # Localize the angle method.
        get_angle = fast_compute_angle

        # Get a reference of the central metal atom.
        # NOTE: This is assumed to be in location 0.
        pt_center = points_xyz[0]

        # Compute the angles among the central
        # metal atom and its closest neighbors.
        for j in range(1, n_closest+1):

            # Get the second (LEFT) point.
            pt_j = points_xyz[j]

            for k in range(j + 1, n_closest+1):

                # Get the third (RIGHT) point.
                pt_k = points_xyz[k]

                # Get the angle in question (in degrees).
                angles_list_append((j, 0, k,
                                    get_angle(pt_j, pt_center, pt_k)))
            # _end_for_

        # _end_for_

        # Return the numpy array (float).
        return np.array(angles_list, dtype=float)
    # _end_def_

    def read_data_file(self, f_path, f_type="CSD"):
        """
        Reads the input data file and returns a list with
        the entries using the PDB file format (see above).

        :param f_path: Path of the metal input file.

        :param f_type: File type CSD (default)/ PDB.

        :return: A list with the required entries.
        """

        # Check the file type.
        if f_type not in {"CSD", "PDB"}:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"The acceptable file types are 'CSD' or 'PDB': {f_type}")
        # _end_if_

        # Create a temporary data structure
        # to hold the data from the file.
        data_list = []

        # Read the data from the file.
        with open(f_path, "r", encoding='utf_8') as f_in:

            # Localize append method for speed.
            data_list_append = data_list.append

            # Process each line.
            for row in f_in.readlines():

                # Convert everything to upper case.
                row = row.upper()

                # Make sure we capture any exceptions
                try:
                    # Get the first entry in the line.
                    # Make sure there are no empty spaces
                    # around the word.
                    entry = str(row[0:6]).strip()

                    # Break when you reach the end of the model.
                    if "END" == entry:
                        break

                    # Process only 'atom' or 'hetero-atom' entries.
                    elif entry in {"ATOM", "HETATM"}:

                        # Create a new row for the dataframe.
                        record = {"ID": int(row[6:11]),
                                  "NAME": str(row[12:16]).strip(),
                                  "X": float(row[30:38]),
                                  "Y": float(row[38:46]),
                                  "Z": float(row[46:54]),
                                  "TYPE": str(row[76:78]).strip()}

                        # Update the list.
                        data_list_append(record)

                    # _end_if_

                except ValueError as v_err:

                    # Show a warning message.
                    print(f"{self.__class__.__name__}: "
                          f"WARNING: {v_err} at {f_path.stem}")

                    # If methods 'int' and 'float' can't convert the strings
                    # they will throw a ValueError. In this case skip and go
                    # to the next row.
                    continue
                # _end_try_

            # _end_for_

        # _end_with_

        # Return the list with the file entries.
        return data_list
    # _end_def_

    @staticmethod
    def find_central_atom(df: pd.DataFrame = None) -> MetalAtom:
        """
        Find which metal atom (from the accepted list)
        is closest to the center of mass.

        :param df: dataframe with the entries from the
        input (metal) PDB file.

        :return: center_metal (as MetalAtom object).
        """

        # Make sure we have an input dataframe.
        if df is None:
            return None
        # _end_if_

        # Find the center mass. This is simply
        # the mean coordinates from all atoms.
        center_mass = df[["X", "Y", "Z"]].mean(axis=0). \
            to_numpy(dtype=float)

        # Center metal and minimum distance.
        center_metal, min_dist = None, np.inf

        # Localize the Euclidean norm function.
        get_euclidean_norm = fast_euclidean_norm

        # Localize numpy method.
        _isfinite = np.isfinite

        # Localize the array function.
        np_array = np.array

        # Scan all the values of the df.
        for it in df.to_dict("records"):

            # Consider only metal atoms from a target list.
            if it["TYPE"].upper().startswith(METAL_TARGETS):

                # Atom coordinates vector.
                it_coord = np_array([it["X"],
                                     it["Y"],
                                     it["Z"]], dtype=float)

                # Distance between coordinates.
                diff = center_mass - it_coord

                # Euclidean norm.
                this_dist = get_euclidean_norm(diff)

                # Update the minimum.
                if _isfinite(this_dist) and this_dist < min_dist:
                    # Update the minimum.
                    min_dist = this_dist

                    # Update the new center atom.
                    center_metal = MetalAtom(**it, DIST=min_dist)
                # _end_if_

            # _end_if_

        # _end_for_

        return center_metal
    # _end_def_

    def estimate_distances(self, f_path, f_type="CSD", max_length=None,
                           n_closest=None):
        """
        This is the main function of the MetalPdbData class. It accepts
        as input a Path(/path/to/the/PDB) and then estimates the sorted
        distances of all the non--hydrogen atoms from the central metal
        atom.

        :param f_path: Path of the (metal) CSD / PDB file.

        :param f_type: File type CSD (default) or PDB.

        :param max_length: Integer that defines the maximum number of
        atoms to consider for the contact map. If 'None' (default) it
        will use all the available atoms.

        :param n_closest: Number of neighbouring points to consider when
        computing the angles.  If 'None' (default), then we will compute
        and return all the angles.

        :return: None.
        """

        # Check file type.
        if f_type not in {"CSD", "PDB"}:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"File type {f_type} is not recognized.")
        # _end_if_

        # If a value has been given.
        if max_length is not None:

            # Check the type.
            if not isinstance(max_length, int):

                # This will raise an error if an input has
                # been given, but it is not an integer type.
                raise TypeError(f"{self.__class__.__name__}: The maximum length for "
                                f"a contact map should be an integer: {max_length}. ")
            # _end_if_

        # _end_if_

        # First we need to  make sure all data are cleared.
        # That means setting all the fields values to None.
        self.reset_all_data()

        # Ensure the f_path is Path.
        f_path = Path(f_path)

        # Make sure is a file.
        if f_path.is_file():

            # STEP: 1
            # Create the dataframe from the raw data file.
            df = pd.DataFrame(data=self.read_data_file(f_path, f_type))

            # Sanity check.
            if df.empty:
                raise RuntimeError(f"{self.__class__.__name__}: "
                                   f"The input data file {f_path} did not return any data.")
            # _end_if_

            # STEP: 2
            # Find the center metal atom.
            center_metal = MetalPdbData.find_central_atom(df)

            # Sanity check.
            if center_metal is None:

                # Display a waring message.
                raise RuntimeError(f"{self.__class__.__name__}: "
                                   f"Center metal not found in {f_path.stem}.")
            # _end_if_

            # Copy the central atom in the object.
            self._center_metal_atom = center_metal

            # Increase the call counter.
            self._stats_counter["n_calls"] += 1

            # Update the statistics dictionary.
            self._stats_counter["metals_freq"][center_metal.TYPE] += 1

            # STEP: 4
            # Get all the distances (ignoring "H" and "C" atoms).
            #

            # Distances list.
            distances = []

            # Localize append method (for faster execution).
            distances_append = distances.append

            # Localize the Euclidean norm function.
            get_euclidean_norm = fast_euclidean_norm

            # Localize numpy method.
            _isfinite = np.isfinite

            # XYZ coordinates of the central metal.
            center_coord = np.array([center_metal.X,
                                     center_metal.Y,
                                     center_metal.Z], dtype=float)

            # Scan all the entries in the dataframe, but
            # consider only non hydrogen or carbon atoms.
            for it in df.to_dict("records"):

                # Skip Hydrogen and Carbon atoms.
                if it["TYPE"] in {"H", "C"}:
                    continue
                # _end_if_

                # Atom coordinates vector.
                it_coord = np.array([it["X"],
                                     it["Y"],
                                     it["Z"]], dtype=float)

                # Distance between coordinates.
                diff = center_coord - it_coord

                # Euclidean norm.
                dist = get_euclidean_norm(diff)

                # Make sure the distance is finite.
                if _isfinite(dist):

                    # Store the pair (id, distance).
                    distances_append((it["ID"], dist))

                # _end_if_

            # _end_for_

            # Sort the list of distances (from small to large).
            sorted_distances = sorted(distances, key=itemgetter(1))

            # Store a copy of the results in the object.
            self._dataset = (df.copy(), np.copy(sorted_distances))

            # Make sure the max_length has been set.
            if max_length is None:

                # Set the max to the same length.
                max_length = len(sorted_distances)

            else:

                # Make sure we are not out of bounds.
                max_length = min(max_length, len(sorted_distances))
            # _end_if_

            # Ensure this var is integer.
            max_length = int(max_length)

            # STEP 5:
            # Use 'cdist' to get the distances among all
            # atoms. This is very fast (using 'C' code).

            # This will 'speed up' the search.
            df.set_index(["ID"], inplace=True)

            # Empty list.
            coord_xyz = []

            # Localize the append method.
            coord_xyz_append = coord_xyz.append

            # Copy the first "max_length" coordinates
            # in a separate list for further process.
            for n, (id_n, _) in enumerate(sorted_distances):

                # Bounds Check: for safety.
                if n >= max_length:
                    break
                # _end_if_

                # Copy the coordinates vector.
                coord_xyz_append(df.loc[id_n][["X", "Y", "Z"]])
            # _end_for_

            # Convert to numpy array.
            coord_xyz = np.array(coord_xyz, dtype=float)

            # Compute distances between each pair of the same
            # array and add the distance map to the object.
            self._distance_map = cdist(coord_xyz, coord_xyz,
                                       metric="euclidean")
            # STEP 6:
            #
            # Optionally get the angles among the 'n_closest' points.
            if self.compute_angles:

                # Store the angles in the object.
                self._angles = self.angles_from_coordinates(coord_xyz,
                                                            n_closest=n_closest)
            # _end_if_

            # STEP 7:
            #
            # Optionally get the convex hull quantities.
            if self.compute_convex_hull:

                # In 3D, we need at least 4 points to define a convex hull.
                if max_length > 3:

                    # If we don't have enough points an error will be raised.
                    try:
                        # Construct a Convex hull from the selected points.
                        conv_hull = ConvexHull(coord_xyz, qhull_options='QJ')

                        # Get the convex hull surface area.
                        self._area = conv_hull.area

                        # Get the convex hull volume.
                        self._volume = conv_hull.volume

                    except QhullError as e:

                        # This error message is not catastrophic. It just
                        # means that we will not have the ConvexHull obj.
                        print(f"{self.__class__.__name__}: Error -> {e}")

                    # _end_try_
                else:
                    # Inform the user we need to increase the max_length.
                    print(f"{self.__class__.__name__}: WARNING: "
                          "You need at least four points (max_length >= 4) "
                          "to define a convex hull in 3D.")
                # _end_if_

            # _end_if_

        # _end_if_

    # _end_def_

    def save_data(self, output_dir=None):
        """
        Save the object's data to an HDF5 file.

        :param output_dir: The directory we want to save the data.
        The default output is the current working directory (CWD).

        :return: None.
        """

        # Check for the output.
        if output_dir is None:
            output_dir = Path.cwd()
        # _end_if_

        # Create an output path.
        path_out = Path(Path(output_dir) / f"metal_{id(self)}.h5")

        # This will hold the fields we are saving to file.
        data_fields = []

        # Create a HDF5 output file (write only).
        with h5py.File(path_out, "w") as f_out:

            # Sanity check.
            if self._area is not None:
                # Update the list.
                data_fields.append("area")

                # Save the convex-hull area.
                f_out.create_dataset("area",
                                     data=self._area)
            # _end_if_

            # Sanity check.
            if self._angles is not None:
                # Update the list.
                data_fields.append("angles")

                # Save the angle values.
                f_out.create_dataset("angles",
                                     data=self._angles)
            # _end_if_

            # Sanity check.
            if self._volume is not None:
                # Update the list.
                data_fields.append("volume")

                # Save the convex-hull volume.
                f_out.create_dataset("volume",
                                     data=self._volume)
            # _end_if_

            # Sanity check.
            if self._distance_map is not None:
                # Update the list.
                data_fields.append("dist_map")

                # Save the distance maps list.
                f_out.create_dataset("dist_map",
                                     data=self._distance_map)
            # _end_if_

            # Sanity check.
            if self._dataset is not None:
                # Unpack the data.
                _, sorted_distances = self._dataset

                # Update the list.
                data_fields.append("sorted_distances")

                # Save the sorted distances.
                f_out.create_dataset("idx_sorted_dist",
                                     data=sorted_distances)
            # _end_if_

        # _end_with_

        # Display info.
        print(f"Saved {data_fields} to: {path_out}")
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This is only a "wrapper" of the
        "estimate_distances" method.
        """
        return self.estimate_distances(*args, **kwargs)
    # _end_def_

    def features_vector(self, map_length=7, num_angles=15):
        """
        Using the pre-computed distance maps and the angles
        construct a "feature vector" according to the input
        requirements.

        If the input requirements are higher than the estimated
        map/angles, the feature vector will be filled with '-1'
        which is the default (N/A) values.

        :param map_length: The size of the map we want to extract.
        Note that the maps are square (MxM).

        :param num_angles: The number of angles we want to include.

        :return: a feature vector with dimensions: 'n*(n-1)/2 + m',
        where n=map_length and m=num_angles.
        """

        # Sanity check.
        if self._distance_map is None:
            raise RuntimeError(f"{self.__class__.__name__}: Distance map is missing.")
        # _end_if_

        # Sanity check.
        if self._angles is None:
            raise RuntimeError(f"{self.__class__.__name__}: Angles list is missing.")
        # _end_if_

        # Local minimum value.
        _min_value = 5

        # Make sure the input parameters are integers.
        map_length = max(_min_value, int(map_length))
        num_angles = max(_min_value, int(num_angles))

        # Preallocate the array with default values.
        new_distance_map = -1.0 * np.ones((map_length,
                                           map_length), dtype=float)

        # Preallocate the array with default values.
        new_angles = -1.0 * np.ones(num_angles, dtype=float)

        # Start copying the angle values.
        for j, value in enumerate(self._angles, start=0):

            # Checking bounds.
            if j > num_angles:
                break
            # _end_if_

            # Get only the last value.
            new_angles[j] = value[-1]
        # _end_for_

        # Get the length of the estimated distance map.
        # Note: The matrix is square (LxL).
        L = self._distance_map.shape[0]

        # Check according to the shapes.
        if L < map_length:

            # If the estimated distance map we will have entries
            # with the default value of '-1'.
            new_distance_map[:L, :L] = self._distance_map.copy()

        elif L > map_length:

            # Here we use only a sub-set of the estimated distance map.
            new_distance_map = self._distance_map[:map_length, :map_length].copy()

        else:

            # Here we have the same dimensions: L == map_length.
            new_distance_map = self._distance_map.copy()
        # _end_if_

        # Augment the total vector (with the unpacked values).
        feature_vector = [*new_distance_map[np.triu_indices(map_length, k=1)],
                          *new_angles]

        # Return the feature vector (as list).
        return feature_vector
    # _end_def_

    def __str__(self):
        """
        Print a readable string presentation of the object.
        This will include its id(), along with its variables.

        :return: a string representation of a MetalPdbData object.
        """

        # Make the output string 1/5.
        str_data_exists = False if self._dataset is None else True

        # Make the output string 2/5.
        str_contact_map = None if self._distance_map is None else self._distance_map.shape

        # Make the output string 3/5.
        str_list_angles = None if self._compute_angles is False else self._angles.shape

        # Make the output string 4/5.
        str_convex_area = None if self._area is None else np.round(self._area, 3)

        # Make the output string 5/5.
        str_convex_vol0 = None if self._volume is None else np.round(self._volume, 3)

        # Return the f-string.
        return f" MetalPdbData Id({id(self)}): {linesep}" \
               f" Dataset exists     = {str_data_exists} {linesep}" \
               f" Contact map shape  = {str_contact_map} {linesep}" \
               f" Angles list shape  = {str_list_angles} {linesep}" \
               f" Convex Hull area   = {str_convex_area} {linesep}" \
               f" Convex Hull volume = {str_convex_vol0} {linesep}" \
               f" Central metal atom = {self._center_metal_atom} {linesep}" \
               f" Statistics         = {self._stats_counter}"
    # _end_def_

# _end_class_
