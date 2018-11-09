from collections import defaultdict

import pandas as pd
import tensorflow as tf
import random as rn
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

class ExhaustiveSearch(object):

    def __init__(self, estimator, score_functions, params_dict, cv_splits):
        """

        :param estimator: estimator object, must implement the sklarn's BaseEstimatro interface
        :param score_functions: dict of score functions to compute
        :param params: dict of lists of the hyperparameter search space. Example: {"p1": [1,2,3], "p2" : [2,2,2]}
        :param cv: iterable of (train_idx, test_idx) splits
        """
        self.estimator = estimator
        self.score_functions = score_functions
        self.params_dict = params_dict
        self.cv_splits = cv_splits
        self.cv_results_ = defaultdict(list)

    def fit(self, X, y, seed=None):
        """
        ATTENTION: this fit method by default reset the seed tf and numpy seeds before each cross validation pass
        begin the search by applying the cv splits to the training data
        :param X:
        :param y:
        :return: pandas dataframe containig the results
        """

        for params in ParameterGrid(self.params_dict):
            np.random.seed(seed)
            rn.seed(seed)
            tf.set_random_seed(seed)
            scores = defaultdict(list)
            # start cross validation
            for (train_idx, test_idx) in self.cv_splits:
                model = clone(self.estimator)
                model.set_params(**params)
                model.fit(X[train_idx], y[train_idx])
                test_probas = model.predict_proba(X[test_idx])[:, 1]
                train_probas = model.predict_proba(X[train_idx])[:, 1]
                for name, scorer in self.score_functions.items():
                    scores["test_" + name].append(scorer(y[test_idx], test_probas))
                    scores["train_" + name].append(scorer(y[train_idx], train_probas))

            avg_scores = {"mean_" + name: np.average(score_list) for name, score_list in scores.items()}
            stds = {"std_" + name: np.std(score_list) for name, score_list in scores.items()}
            [self.cv_results_[param_name].append(param_value) for param_name, param_value in params.items()]
            [self.cv_results_[mean_score].append(mean_value) for mean_score, mean_value in avg_scores.items()]
            [self.cv_results_[std_score].append(std_value) for std_score, std_value in stds.items()]