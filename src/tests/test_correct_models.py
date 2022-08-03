"""
    Verify that we use the correct trained models. This test should
    FAIL if we use re-trained models, and it will have to be updated.
"""

import unittest
import joblib
from pathlib import Path

class TestTrainedModels(unittest.TestCase):

    def test_correct_hash(self):
        """
        Test that the MD5 hash-codes of the loaded models
        are the ones that we have distributed.

        :return: None.
        """

        # Correct md5-hash codes.
        model_hash = {"CSD": 'e822834883d8f6a3546480f5ae77f5e1',
                      "PDB": '5e26fd38b2d0628a000f1202ab3511ee'}

        # Get the hash code of the CSD_CSD model.
        CSD_hash = joblib.hash(joblib.load(Path("../../models/HPO_CSD_CSD_CV.model")),
                               hash_name='md5')

        # Check against the correct md5-hash.
        self.assertEqual(CSD_hash, model_hash["CSD"],
                         msg="CSD model md5-hash code is not correct.")

        # Get the hash code of the PDB_PDB model.
        PDB_hash = joblib.hash(joblib.load(Path("../../models/HPO_PDB_PDB_CV.model")),
                               hash_name='md5')

        # Check against the correct md5-hash.
        self.assertEqual(PDB_hash, model_hash["PDB"],
                         msg="PDB model md5-hash code is not correct.")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
