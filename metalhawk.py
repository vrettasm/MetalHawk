#!/usr/bin/env python3 -O

import sys
import argparse

# Check the current python version before running.
if sys.version_info < (3, 8, 0):
    sys.exit("Error: MetalHawk requires Python 3.8 or greater.")
# _end_if_

import os
import textwrap
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


from src.metal_auxiliaries import CLASS_TARGETS
from src.model_predictor import MetalSitesPredictor

# INFO:
__program__ = "MetalHawk"
__version__ = '0.1.0'
__author__ = 'Michail Vrettas, PhD'
__email__ = 'vrettasm@duck.com'


# Main function.
def main(input_file=None, csd_target_model=True, output_path=None, verbose=False):
    """
    Main function that wraps the call of the predict method. First we create
    a MetalSitesPredictor object. We assume that the trained model is located
    in a folder named "/models/" that exists inside the parent directory (i.e.
    the same directory where the current script is stored).

    :param input_file: This is the input (PDB format) file that we want to predict
    the metal site class.

    :param csd_target_model: This is the target model we want to use. CSD (default)
    or PDB. Note that if we don't use the correct target model, the prediction will
    still predict values.

    :param output_path: The directory (path) where we want to store the output
    file.

    :param verbose: This is a boolean flag that determines whether we want to
    display additional information on the screen. Default is "False".

    :return: None.
    """

    try:
        # Get the parent folder of the module.
        parent_dir = Path(__file__).resolve().parent

        # Make sure the model directory is Path.
        model_dir = Path(parent_dir/"models/")

        # Sanity check.
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory {model_dir} doesn't exist.")
        # _end_if_

        # Construct the path of the model we want to load.
        if csd_target_model:

            # CSD -> CSD (Bayesian optimized model).
            target_path = Path(model_dir/"HPO_CSD_CSD_CV.model")
        else:

            # PDB -> PDB (Bayesian optimized model).
            target_path = Path(model_dir/"HPO_PDB_PDB_CV.model")
        # _end_if_

        # Create a predictor object.
        nn_predict = MetalSitesPredictor(dir_model=target_path)

        # Count the successful predictions.
        count_success = 0

        # Store the prediction results.
        results = {"file": [], "prediction": [], "entropy": []}

        # Check for verbose.
        if verbose:

            # In verbose mode, use the list without 'tqdm'.
            file_range = input_file
        else:

            # In the standard mode, use 'tqdm'.
            file_range = tqdm(input_file, " Predicting metal sites ... ")
        # _end_if_

        # Process all input files.
        for it, f_in in enumerate(file_range):

            # Make sure is a Path.
            f_path = Path(f_in)

            # If the file exists.
            if f_path.is_file():

                # Make the predictions.
                class_i, entropy_i = nn_predict(f_path)

                # Store the results in the final dictionary.
                results["file"].append(f_path.stem)
                results["entropy"].append(entropy_i)
                results["prediction"].append(CLASS_TARGETS[class_i])

                # Increase counter.
                count_success += 1

                # Check for verbosity.
                if verbose:

                    # This might clatter the screen output if there are many files!
                    print(f" {it}: File= {f_path.stem},"
                          f" Prediction= {CLASS_TARGETS[class_i]},"
                          f" Entropy= {entropy_i:.4E} \n")
                # _end_if_

            else:
                raise FileNotFoundError(f"File {f_path} not found.")
            # _end_if_

        # _end_for_

        # Convert results to DataFrame.
        df = pd.DataFrame(results)

        # Print the results.
        print(f"\n {df}")

        # Final message.
        if verbose:

            # Print the total predictions.
            print(f" Successfully predicted {count_success} file(s)")
        # _end_if_

        # Check if we want to save the results in a csv file.
        if output_path:

            # Make a header for the filename.
            header = "CSD" if csd_target_model else "PDB"

            # Make sure it is Path object.
            output_path = Path(output_path)

            # Get the timestamp.
            date_now = datetime.now().strftime("%Y_%m_%d_%I_%M_%S")

            # Save to output_path with a date-related name.
            df.to_csv(Path(output_path/f"{header}_metalhawk_predictions_{date_now}.csv"))
        # _end_if_

    except Exception as e1:

        # Exit the program.
        sys.exit(f" Program ended with message: {e1}")
    # _end_try_

# _end_main_


# Run the main script.
if __name__ == "__main__":

    # Check if we have given input parameters.
    if len(sys.argv) > 1:

        # Create a parser object.
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent('''
                                         
                                         [MetalHawk]
                                         
                                         Python metal sites predictor, with artificial neural networks (ANN).
                                         
                                         We have trained (and optimized) two different models. One on CSD metal
                                         sites and a second one on PDB metal sites. By default the program will
                                         choose the CSD model to use for prediction, unless we pass explicitly
                                         the option '--no-csd'.
                                                                                  
                                         The selected model can predict one of the following target classes:
                                         -------------------------------------------------------------------
                                            0: LIN (Linear)
                                            1: TRI (Trigonal planar)
                                            2: TET (Tetrahedral)
                                            3: SPL (Square planar)
                                            4: SQP (Square pyramidal)
                                            5: TBP (Trigonal bi-pyramidal)
                                            6: OCT (Octahedral)
                                         
                                         Along with the predicted class the program provides the "entropy" value,
                                         as a measure of uncertainty. Note that for seven classes, assuming equal
                                         priors, the maximum value of entropy is equal to: '1.945910149055313'.
                                         '''))

        # Input (PDB) file with the residue coordinates.
        parser.add_argument("-f", "--file", type=str, nargs='+', required=True,
                            help="Input file(s) (Path/String).")

        # Add the target model format (True=CSD).
        parser.add_argument("--csd", dest="csd_model", action="store_true",
                            help="The default target model is the 'CSD'.")

        # Add the alternative target model format (False=PDB).
        parser.add_argument("--no-csd", dest="csd_model", action="store_false",
                            help="Alternatively we set the target model to 'PDB'.")

        # Output path to save the predictions.
        parser.add_argument("-o", "--out", type=str, default=None,
                            help="Output 'path' to save the predicted values. ")

        # Enables verbosity.
        parser.add_argument("--verbose", dest="verbose", action="store_true",
                            help="Display information while running.")

        # Shows the version of the program.
        parser.add_argument("--version", action="version",
                            version=f" {__program__} (c), version: {__version__}",
                            help="Print version information and exit.")

        # Make sure the defaults are set.
        parser.set_defaults(csd_model=True, verose=False, out=None)

        # Parse the arguments.
        args = parser.parse_args()

        # Call the main function.
        main(input_file=args.file, csd_target_model=args.csd_model,
             output_path=args.out, verbose=args.verbose)

    else:
        # Display error message.
        sys.exit(f"Error: Not enough input parameters. {os.linesep}"
                 f" Run : {sys.argv[0]} -h/--help ")
    # _end_if_

# _end_program_
