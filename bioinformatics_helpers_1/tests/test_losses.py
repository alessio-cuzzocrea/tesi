#import unittest
#
#import numpy as np
#import tensorflow as tf
#from bioinformatics_helpers.utils import hingesig_tf
##from bioinformatics_helpers.utils import hingesig
#from scipy.special import logit
#
#
#class LossesTest(unittest.TestCase):
#    def test_hingesig_loss(self):
#        def np_hinge(y_true, y_pred):
#            transform_y_true = np.vectorize(lambda x: -1 if x == 0 else x)
#            clipped_y_pred = np.clip(y_pred, 1e-20, 1)
#            compl_y_pred = np.clip(np.subtract(1, y_pred), 1e-20,1)
#            logit = np.log2(clipped_y_pred) - np.log2(compl_y_pred)
#            return np.mean(np.maximum(np.subtract(1, transform_y_true(y_true) * logit),0))
#        
#        y_pred = [0.6, 0.3, 1]
#        y_true = [1.0,1.0,-1.0]
#        self.assertAlmostEqual(np_hinge(y_true, y_pred), hingesig(y_true, y_pred).eval(), delta=0.9)
#
#    def test_theano_vs_tf(self):
#        y_pred = [0.9, 0.01, 0.5, 0.2, 0.1, 0.8, 0.3]
#        y_true = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1]
#        sess = tf.Session()
#        logits = logit(y_pred).tolist()
#        with sess.as_default():
#            self.assertAlmostEqual(
#                hingesig(y_true, y_pred).eval().tolist(),  
#                hingesig_tf(y_true, y_pred).eval().tolist(),
#                delta=0.005
#               )
#if __name__ == '__main__':
#    unittest.main()
