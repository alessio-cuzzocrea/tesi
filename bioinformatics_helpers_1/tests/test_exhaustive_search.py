import unittest
from itertools import product

from bioinformatics_helpers.utils import ExhaustiveSearch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd

# Neither of the following two estimators inherit from BaseEstimator,
# to test hyperparameter search on user-defined classifiers.
class MockClassifier(object):
    """Dummy classifier to test the parameter search algorithms"""
    def __init__(self, foo_param=0):
        self.params = {}

    def fit(self, X, Y):
        return self

    def predict_proba(self, T):
        positive_class = np.array([i % 2 for i in range(0, len(T))])
        negative_class = 1 - positive_class
        return np.array([negative_class, positive_class])

    def get_params(self, deep=False):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([0, 1, 0, 1])

param_grid = {
    "np_ratio" : [3],
    "positive_sample_perc" : [2],
    "negative_perc" : [3,4]
}
score_functions ={
    "AU_ROC" : roc_auc_score,
    "AVG_PREC" : average_precision_score
}


class TestExhaustiveSearch(unittest.TestCase):

    def test_fit(self):
        train_idx = np.arange(0, np.int32(len(X)/2))
        test_idx = np.arange(np.int32(len(X)/2), len(X))
        grid = ExhaustiveSearch(MockClassifier(), score_functions, param_grid, [(train_idx, test_idx)] * 2)
        grid.fit(X,y)
        n_param_combinations = 2
        cv_results = pd.DataFrame(grid.cv_results_)
        self.assertEqual(n_param_combinations, len(cv_results))
        self.assertTrue(np.all(np.array(cv_results["mean_test_AU_ROC"]) == 1.0))
        self.assertTrue(np.all(np.array(cv_results["mean_test_AVG_PREC"]) == 1.0))


if __name__ == '__main__':
    unittest.main()
