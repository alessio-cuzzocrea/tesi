from keras.utils import Sequence
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.base import clone


def input_consistency_check(y_set, x_set, positive_sample_perc, np_ratio, negative_perc):
    """ This functioncheck the consistency of the input
    
    Arguments:
        y_set {np.array} -- the set of the labels
        x_set {np.array} -- the set of the samples
        positive_sample_perc {float} -- a float greater than 0.
        np_ratio {float} -- a float greater than 0
        negative_perc {float} -- a float between (0,1]
    """
    if len(np.unique(y_set)) != 2:
        raise AttributeError("the target must be binary")
    if len(x_set) != len(y_set):
        raise AttributeError("x and y must have the same number of rows")
    if positive_sample_perc <= 0.0:
        raise AttributeError("positive_sample_perc must greater than 0")
    if np_ratio <= 0:
        raise AttributeError("np_ratio must be greater than 0")
    if negative_perc > 1 or negative_perc <= 0:
        raise AttributeError("negative_perc must be greater than 0 and lower than 1")




class BalancedGenerator(Sequence):
"""This class generate balanced minibatches for a binary classification task.
    The negative class here is intended to be 0, the positive 1.
    
Arguments:
    Sequence  -- Keras base class for all generators.
"""
    def __init__(self, x_set, y_set, positive_sample_perc=1.0,np_ratio=1.0, negative_perc=1.0):
        """Constructor
        
        Arguments:
            x_set {np.ndarray} -- the dataset
            y_set {np.ndarray} -- the labels
        
        Keyword Arguments:
            positive_sample_perc {float} --  (default: {1.0})
                This number is used to get the number of positive samples to include in the next batch.
                So len(positive_samples) * positive_sample_perc are included in the next batch.
                If positive_sample_perc >= 1: then a bootstrap sampling is used
                If positive_sample_perc < 1: a random undersapling is done, without bootstrap.
            np_ratio {float} -- (default: {1.0})
                negative/positive ratio. After evaluating how much positives include in the next batch(say k), 
                np_ratio * k negatives are picked.
            negative_perc {float} -- (default: {1.0})
                whether to randomly undersample the negative data. If 0 no negatives will be picked, if 1 no undersampling is done.
        """
        input_consistency_check(y_set, x_set, positive_sample_perc, np_ratio, negative_perc)

        self.x = np.copy(x_set)
        self.y = np.copy(y_set)
        self.positive_sample_perc = positive_sample_perc
        self.np_ratio = np_ratio
        self.negative_perc = negative_perc
        # get number of positive samples
        self.n_positives = len(self.y[self.y == 1])
        self.minibatch_positives = np.int32(np.ceil(self.n_positives * self.positive_sample_perc))
        # get number of negative samples
        self.minibatch_negatives = np.int32(np.ceil(self.np_ratio * self.minibatch_positives))
        self.positive_indexes = np.copy(np.where(self.y == 1)[0])
        self.negative_indexes = np.copy(np.where(self.y == 0)[0])
        # slice negative index to get the perc requested
        if negative_perc != 1:
            n_negatives = len(self.negative_indexes)
            choice_size = np.int32(np.ceil(n_negatives * negative_perc))
            self.negative_indexes = np.random.choice(self.negative_indexes, choice_size, replace=False)

    def on_epoch_end(self):
        np.random.shuffle(self.negative_indexes)

    def __len__(self):
        # the number of minibatches depends on the negatives
        n_negatives = len(self.y[self.y == 0])
        return np.int32(np.ceil((self.negative_perc * n_negatives) / self.minibatch_negatives))
    
    def __getitem__(self, idx):
        replace = self.positive_sample_perc >= 1
        sampled_positive_indexes = np.random.choice(self.positive_indexes, self.minibatch_positives, replace)
        
        sampled_negative_indexes = self.negative_indexes[(idx * self.minibatch_negatives):(idx + 1) * self.minibatch_negatives]
        batch_x = np.concatenate([
            self.x[sampled_positive_indexes],
            self.x[sampled_negative_indexes]
        ])
        batch_y = np.concatenate([
            self.y[sampled_positive_indexes],
            self.y[sampled_negative_indexes]
        ])
        return np.array(batch_x), np.array(batch_y)
