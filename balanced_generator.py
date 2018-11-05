from keras.utils import Sequence
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.base import clone


def input_consistency_check(y_set, x_set, positive_sample_perc, np_ratio, negative_perc):
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

"""
    This class generate balanced minibatches whene the positive class is the minority one
    the negative class here is intended to be 0, the positive 1
    
"""


class BalancedGenerator(Sequence):

    def __init__(self, x_set, y_set, positive_sample_perc=1.0,np_ratio=1.0, negative_perc=1.0):
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



class SyntheticGenerator(Sequence):

    def __init__(self, x_set, y_set, negative_batch_size=5000, np_ratio=1.0, n_shuffle=True):
        if len(np.unique(y_set)) != 2:
            raise AttributeError("the target must be binary")
        if len(x_set) != len(y_set):
            raise AttributeError("x and y must have the same number of rows")
        self.x_positives = np.copy(x_set[y_set == 1])
        self.x_negatives = np.copy(x_set[y_set == 0])
        self.y_positives = np.copy(y_set[y_set == 1])
        self.y_negatives = np.copy(y_set[y_set == 0])
        self.np_ratio=np_ratio
        self.batch_size = negative_batch_size
        self.shuffle = n_shuffle
        # shuffle data, if requested
        self._shuffle_negatives()

    def _shuffle_negatives(self):
        self.negative_indexes = np.array(np.arange(0, len(self.x_negatives)))
        if self.shuffle:
            np.random.shuffle(self.negative_indexes)

    def on_epoch_end(self):
        self._shuffle_negatives()
    
    def __len__(self):
        # the number of minibatches depends on the negatives
        n_negatives = len(self.y_negatives)
        return np.int32(np.ceil(n_negatives / self.batch_size))

    def __getitem__(self, idx):
        batch_negative_indexes = self.negative_indexes[(idx * self.batch_size):(idx + 1) * self.batch_size]
        batch_negatives = self.x_negatives[batch_negative_indexes]
        batch_negatives_labels = self.y_negatives[batch_negative_indexes]
        n_positives = self.batch_size / self.np_ratio
        over_sampler = ADASYN(ratio={1:n_positives})
        resampled_X, resampled_y = over_sampler.fit_sample(
            np.concatenate((batch_negatives, self.x_positives)),
            np.concatenate((batch_negatives_labels, self.y_positives))
            )
        return resampled_X, resampled_y







"""
    fake class, not meant to be used!
"""
class NormalGenerator(Sequence):

    def __init__(self, x_set, y_set, positive_sample_perc=1.0,np_ratio=1.0, negative_perc=1.0):
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

        self.x = x_set
        self.y = y_set
        self.positive_sample_perc = positive_sample_perc
        self.np_ratio = np_ratio
        self.negative_perc = negative_perc
        # get number of positive samples
        self.n_positives = len(self.y[self.y == 1])
        self.minibatch_positives = np.int32(np.ceil(self.n_positives * self.positive_sample_perc))
        # get number of negative samples
        self.minibatch_negatives = np.int32(np.ceil(self.np_ratio * self.minibatch_positives))
        self.positive_indexes = np.where(self.y == 1)[0]
        self.negative_indexes = np.where(self.y == 0)[0]
        # slic negative index to get the perc requested
        n_negatives = len(self.negative_indexes)
        self.negative_indexes = \
            self.negative_indexes[0:np.int32(np.ceil(self.negative_perc * n_negatives))]

    def __len__(self):
        # the number of minibatches depends on the negatives
        n_negatives = len(self.y[self.y == 0])
        return np.int32(np.ceil((self.negative_perc * n_negatives) / self.minibatch_negatives))

    def __getitem__(self, idx):
        replace = self.positive_sample_perc >= 1
        sampled_positive_indexes = self.positive_indexes

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
