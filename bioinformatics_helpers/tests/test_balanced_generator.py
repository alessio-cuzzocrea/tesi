import unittest

import numpy as np

import glob
import json
import platform
from bioinformatics_helpers.balanced_generator import BalancedGenerator
from bioinformatics_helpers.balanced_generator import SyntheticGenerator
np.random.seed(20)
class CustomAssertions:
    def assertDatasetContainsBatch(self, dataset,batch):
        try:
            for sample in batch:
                dataset.index(sample)
        except ValueError:
            raise AssertionError("Batch is not contained in dataset", dataset, batch)


class BalancedGeneratorTest(unittest.TestCase, CustomAssertions):
    def get_dataset(self,n_samples, n_positives):
        X = np.arange(0, n_samples)
        y = np.concatenate([
            np.ones(n_positives, dtype=np.int32),
            np.zeros(n_samples - n_positives, dtype=np.int32)
        ])
        return X, y

    def test_len(self):
        X,y = self.get_dataset(n_samples=30, n_positives=5)

        gen = BalancedGenerator(X, y, positive_sample_perc=0.1,np_ratio=1, negative_perc=1)
        self.assertEqual(gen.__len__(), 25)

    def test_bigger_len(self):
        X,y = self.get_dataset(n_samples=9888, n_positives=10)

        gen = BalancedGenerator(X, y, positive_sample_perc=0.5, np_ratio=1, negative_perc=1)
        self.assertEqual(gen.__len__(), np.ceil(9878/5))

    def test_bootstrapping_len(self):
        n_samples = 1000000
        n_positives = 300
        X,y = self.get_dataset(n_samples, n_positives)

        gen = BalancedGenerator(X, y, positive_sample_perc=1, np_ratio=1, negative_perc=1)
        self.assertEqual(gen.__len__(), np.ceil((n_samples - n_positives) / 300))

    def test_oversampling_len(self):
        n_samples = 1000000
        n_positives = 300
        X,y = self.get_dataset(n_samples, n_positives)

        gen = BalancedGenerator(X, y, positive_sample_perc=1.6, np_ratio=1, negative_perc=1)
        self.assertEqual(gen.__len__(), np.ceil((n_samples - n_positives) / 480))

    def test_oversampling_len(self):
        n_samples = 1000000
        n_positives = 300
        X,y = self.get_dataset(n_samples, n_positives)

        gen = BalancedGenerator(X, y, positive_sample_perc=1.6, np_ratio=1, negative_perc=1)
        self.assertEqual(gen.__len__(), np.ceil((n_samples - n_positives) / 480))

    def test_negative_perc_len(self):
        n_samples = 1000000
        n_positives = 300
        X, y = self.get_dataset(n_samples, n_positives)

        gen = BalancedGenerator(X, y, positive_sample_perc=0.6, np_ratio=1, negative_perc=0.7)
        self.assertEqual(gen.__len__(), np.ceil(((n_samples - n_positives)*0.7) / 180))

    def test_get_item(self):
        n_samples = 100
        n_positives = 10
        X, y = self.get_dataset(n_samples, n_positives)
        dataset = list(zip(X,y))
        gen = BalancedGenerator(X, y, positive_sample_perc=1, np_ratio=1, negative_perc=1)
        batch_X, batch_y = gen.__getitem__(0)
        batch = list(zip(batch_X, batch_y))
        self.assertDatasetContainsBatch(dataset, batch)


    def test_get_all_items(self):
        print("started test")
        n_samples = 100
        n_positives = 10
        X, y = self.get_dataset(n_samples, n_positives)
        dataset = list(zip(X, y))
        gen = BalancedGenerator(X, y, positive_sample_perc=1, np_ratio=1, negative_perc=0.5)
        for i in range(gen.__len__()):
            batch_X, batch_y = gen.__getitem__(i)
            batch = list(zip(batch_X, batch_y))
            self.assertDatasetContainsBatch(dataset, batch)


class SyntethicGeneratorTest(unittest.TestCase, CustomAssertions):
    def get_dataset(self,n_samples, n_positives):
        X = np.arange(0, n_samples)
        y = np.concatenate([
            np.ones(n_positives, dtype=np.int32),
            np.zeros(n_samples - n_positives, dtype=np.int32)
        ])
        return X, y

    def test_len(self):
        #dataset with 4800 negatives and 200 postives
        X,y = self.get_dataset(n_samples=5000, n_positives=200)
        generator = SyntheticGenerator(X,y,100)
        self.assertEqual(generator.__len__(), 48)

        generator = SyntheticGenerator(X,y,345)
        self.assertEqual(generator.__len__(), 14)

    def test_balanced_minibatch(self):
        X,y = self.get_dataset(n_samples=100, n_positives=10)
        generator = SyntheticGenerator(X.reshape(-1,1),y,50)

        X_batch, y_batch = generator.__getitem__(0)

        self.assertAlmostEqual(len(X_batch[y_batch==0]), len(X_batch[y_batch==1]), delta=5)
        self.assertEqual(len(X_batch[y_batch == 0]), 50)
        (_, idx_X, idx_X_batch) = np.intersect1d(X, X_batch, return_indices=True)
        np.testing.assert_array_equal(y[idx_X], y_batch[idx_X_batch])

    def test_balanced_minibatch_np_ratio(self):
        X,y = self.get_dataset(n_samples=100, n_positives=10)
        generator = SyntheticGenerator(X.reshape(-1,1),y,50, np_ratio=5)

        X_batch, y_batch = generator.__getitem__(0)

        self.assertAlmostEqual(len(X_batch[y_batch==0]), 5*len(X_batch[y_batch==1]), delta=5)
        self.assertEqual(len(X_batch[y_batch == 0]), 50)
        self.assertEqual(len(X_batch[y_batch == 1]), 10)
        (_, idx_X, idx_X_batch) = np.intersect1d(X, X_batch, return_indices=True)
        np.testing.assert_array_equal(y[idx_X], y_batch[idx_X_batch])

if __name__ == '__main__':
    unittest.main()
