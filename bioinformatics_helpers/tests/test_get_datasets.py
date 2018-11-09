import unittest
from bioinformatics_helpers.utils import get_mendelian_dataset
import numpy as np
import pandas as pd

class TestGetDatasets(unittest.TestCase):
    def test_get_mendelian_dataset_as_np_matrix(self):
        train_X, train_Y, test_X, test_y = get_mendelian_dataset()
        self.assertIsInstance(train_X, np.ndarray)
        self.assertIsInstance(test_X, np.ndarray)
        self.assertEqual(train_X.shape, (981388, 26), msg="train shape is not correct")
        self.assertEqual(test_X.shape, (19018, 26), msg="test shape is not correct")

    def test_get_mendelian_dataset_as_np_df(self):
        train_X, train_Y, test_X, test_y = get_mendelian_dataset(return_df=True)
        self.assertIsInstance(train_X, pd.DataFrame)
        self.assertIsInstance(test_X, pd.DataFrame)
        self.assertEqual(train_X.shape, (981388, 26), msg="train shape is not correct")
        self.assertEqual(test_X.shape, (19018, 26), msg="test shape is not correct")



if __name__ == '__main__':
    unittest.main()
