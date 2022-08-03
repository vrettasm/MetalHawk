"""
This module includes the main interface that handles the predictions.
"""

# Python import(s).
from pathlib import Path

import joblib
import warnings
import numpy as np

# Custom code imports.
from src.metal_pdb_data import MetalPdbData
from src.metal_auxiliaries import fast_entropy


# Predictor Class.
class MetalSitesPredictor(object):
    """
    Metal Sites Predictor class. This is the "wrapper" class that performs the
    predictions in the input vectors. It processes the input (PDB) files, extracts
    the features and finally performs the prediction using pre-trained models.
    """

    # Object variables.
    __slots__ = ("nn_model", "dir_model", "metal_tool")

    # Constructor.
    def __init__(self, dir_model=None):
        """
        Constructs an object that will perform the metal sites prediction.

        :param dir_model: (Path) Directory where the trained ANN model exist.
        """

        # Make sure the target is Path.
        self.dir_model = Path(dir_model)

        # Make sure the model exists.
        if not self.dir_model.is_file():
            raise FileNotFoundError(f"{self.__class__.__name__}: "
                                    f"Target model doesn't exist: {self.dir_model}.")
        # _end_if_

        # Load the (target) neural network model.
        self.nn_model = joblib.load(self.dir_model)

        # Get the hash of the loaded model.
        hash_code = joblib.hash(self.nn_model, hash_name='md5')

        # Check the 'md5-hash code' of the loaded files.
        # CSD_CSD model: e822834883d8f6a3546480f5ae77f5e1
        # PDB_PDB model: 5e26fd38b2d0628a000f1202ab3511ee
        if hash_code not in ('e822834883d8f6a3546480f5ae77f5e1',
                             '5e26fd38b2d0628a000f1202ab3511ee'):

            # Show a warning message instead of an error.
            warnings.warn("Unknown MD5-hash code for the loaded model.")

        # _end_if_

        # For parsing the PDB/CSD files.
        self.metal_tool = MetalPdbData(compute_angles=True)
    # _end_def_

    @property
    def model_path(self):
        """
        Accessor (getter) of the model input path.

        :return: dir_model.
        """
        return self.dir_model

    # _end_def_

    def predict(self, f_path):
        """
        Primary method of a "MetalSitesPredictor" class. It accepts a PDB file
        as input, constructs the input vector and passes it to the trained NN.

        :param f_path: (string) PDB file with the residue / atom coordinates.

        :return: the predicted (with the highest probability) class and its
        entropy value.
        """

        # Make sure the input file is a Path.
        f_path = Path(f_path)

        # Sanity check.
        if not f_path.is_file():
            raise FileNotFoundError(f"{self.__class__.__name__} : "
                                    f"File {f_path} doesn't exist.")
        # _end_if_

        # Run the analysis. With fixed settings.
        self.metal_tool(f_path, max_length=8, n_closest=6)

        # Create the feature vector.
        x_test = np.array(self.metal_tool.features_vector())

        # Get the prediction probabilities.
        y_probs = self.nn_model.predict_proba(x_test.reshape(1, -1))

        # Remove the singleton dimension. It makes
        # the rest of the function calls faster.
        y_probs = y_probs.squeeze()

        # Compute the entropy (using the probabilities).
        y_entropy = fast_entropy(y_probs)

        # Return the class with the highest probability,
        # and it's entropy value.
        return np.argmax(y_probs), np.maximum(y_entropy, 0.0)
    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is only a "wrapper" method of the
        predict() method.
        """
        return self.predict(*args, **kwargs)
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the
        object. This will include its id(), along with its field
        values.

        :return: a string representation of a MetalSitesPredictor
        object.
        """

        return f" MetalSitesPredictor Id({id(self)}): \n" \
               f" Model path: {self.dir_model} \n" \
               f" Model pipe: {self.nn_model}"
    # _end_def_

# _end_class_
