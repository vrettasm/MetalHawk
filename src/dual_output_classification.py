import numpy as np
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class DualOutputClassifier(object):
    """
    This class implements a "dual-output" classifier where the model
    is trained to predict one target first, and then use this output
    to condition the prediction of the second class. In practice, we
    train two different classifiers!

    The procedure works as follows:
        1) The first classifier is trained on the "first-k" features
        of the input matrix [x] to predict "Coordinate Numbers" (CN).
        These include only the distances from the contact map.

        2) The second classifier is trained on all features [x + CN]
        of the augmented matrix to predict "Geometric Classes" (GC).
        These include the distances from the contact map + the angles
        around the central metal atom + the estimated CN (1st classifier).
    """

    # Default constructor.
    def __init__(self, method=None, add_scaler=False, *args, **kwargs):
        """
        Initializes the class object with the all the parameters that we
        will pass to the estimators.

        :param method: any scikit estimator method.

        :param add_scaler: include data StandardScaler in the pipelines.

        :param args: these are the arguments that we want to pass to the
        model's __init__ method.

        :param kwargs: these are the key-worded arguments that we want to
        pass to the model's __init__ method.
        """
        # Get a reference of the arguments.
        self.args = args

        # Get a reference of the keyword arguments.
        self.kwargs = kwargs

        # Classifier placeholder.
        self.classifiers = None

        # Accept the estimator.
        self.method = method

        # Add a data scaler.
        self.add_scaler = add_scaler

        # Number of features for the first classifier.
        self.k = None
    # _end_def_

    # Mandatory function.
    def fit(self, x, y, k=None):
        """
        This method will call sequentially the fit() methods
        of all the classifiers.

        :param x: input numpy array (2D).

        :param y: output (targets) numpy array (2D).

        :param k: number of features for the first classifier.

        :return: self.
        """
        # Make sure the inputs/targets are 2 dimensional.
        x, y = map(np.atleast_2d, (x, y))

        # Sanity check.
        if x.shape[0] != y.shape[0]:
            # Cancel the fit.
            self.k = None

            # Raise the error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               "Input dimensions mismatch.")
        # _end_if_

        # Sanity check.
        if y.shape[1] != 2:
            # Cancel the fit.
            self.k = None

            # Raise the error.
            raise RuntimeError(f"{self.__class__.__name__}: "
                               "Output dimensions mismatch.")
        # _end_if_

        # Sanity check.
        if k is None:

            # In this case use all the input features.
            k = x.shape[1]
        else:

            # Make sure we are not out of bounds.
            k = np.minimum(k, x.shape[1])
        # _end_if_

        # Ensure correct type.
        k = int(k)

        # Store it in the object to use in the predictions.
        self.k = k

        # Stores (locally in class) the two classifiers.
        self.classifiers = []

        # Start building the steps list.
        clf_steps = []

        # The data Scaler is not applied in all methods.
        if self.add_scaler:
            clf_steps.append(('RobustScaler',
                              RobustScaler(copy=True)))
        # _end_if_

        # Add the estimation method in the end.
        clf_steps.append((f'{self.method.__name__}',
                          self.method(*self.args, **self.kwargs)))

        # NOTE: Get a "DEEPCOPY" of the clf_steps otherwise the second
        # fit will be referenced in the first pipe, and we will get an
        # error in the prediction function.

        # Create a new pipeline for the "first" target.
        pipe_1 = Pipeline(steps=deepcopy(clf_steps), verbose=True)

        # Create a new pipeline for the "second" target.
        pipe_2 = Pipeline(steps=deepcopy(clf_steps), verbose=True)

        # Information about 1-st classifier.
        print("\n >>> Fitting 1st classifier ... ")

        # Use the first-k input features of array x for
        # the prediction of the coordinates number (CN).
        self.classifiers.append(pipe_1.fit(x[:, :k], y[:, 1]))

        # Information about 2-nd classifier.
        print("\n >>> Fitting 2nd classifier ... ")

        # Use the all input features of 'x + CN' for
        # the prediction of the geometry classes (GM).
        self.classifiers.append(pipe_2.fit(np.hstack([x, y[:, 1:]]), y[:, 0]))

        # Return the object itself.
        return self

    # _end_def_

    # Mandatory function.
    def predict(self, x):
        """
        This function will call the two predict() functions,
        of the pre-fitted predictors, with a specific order.

        :param x: test input variables.

        :return: the predictions 'y_predict'.
        """

        # Sanity check.
        if self.k is None:
            raise RuntimeError(f"{self.__class__.__name__}: The model is not fit yet.")
        # _end_if_

        # Make sure the dimensions are 2D.
        x = np.atleast_2d(x)

        # Create the (return) predictions array.
        y_predict = np.zeros((x.shape[0], 2), dtype=float)

        # Get the prediction from the '1st' pipeline.
        y_predict[:, 1] = self.classifiers[0].predict(x[:, :self.k])

        # Augment the input array 'x' with the
        # prediction from the first classifier.
        z = np.hstack([x, y_predict[:, 1:]])

        # Get the prediction from the '2nd' pipeline.
        y_predict[:, 0] = self.classifiers[1].predict(z)

        # Return the total (dual) predictions.
        return y_predict
    # _end_def_

    # Mandatory function.
    def predict_proba(self, x):
        """
        This function will call the two predict() functions,
        of the pre-fitted predictors, with a specific order.

        :param x: test input variables.

        :return: predictions probabilities 'y_predict_proba'.
        """
        # Make sure the dimensions are 2D.
        x = np.atleast_2d(x)

        # Get the predictions from the '1st' pipeline.
        y_predict = self.classifiers[0].predict(x[:, :self.k])

        # Augment the input array 'x' with the
        # prediction from the first classifier.
        z = np.hstack([x, y_predict])

        # Create the (return) predictions array.
        y_predict_proba = np.zeros((x.shape[0], 2), dtype=float)

        # Get the prediction probabilities from the '1st' pipeline.
        y_predict_proba[:, 1] = self.classifiers[0].predict_proba(x[:, :self.k])

        # Get the prediction probabilities from the '2nd' pipeline.
        y_predict_proba[:, 0] = self.classifiers[1].predict_proba(z)

        # Return the total (dual) prediction probabilities.
        return y_predict_proba
    # _end_def_

# _end_class_
