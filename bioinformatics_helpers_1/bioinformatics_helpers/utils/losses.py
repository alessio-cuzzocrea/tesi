import theano.tensor as T
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import numpy as np


def hingesig(y_true, y_pred):
    """Computes the hingeles for a sigmoidal output by apply the logit to y_pred.
       Note: this function is intended for THEANO.
    Arguments:
        y_true  -- a theano tensor holding the true labels
        y_pred -- a theano tensor holding the raw pradictions, i.e. the sigmoid output
    
    Returns:
        theano tensor with hingelosss
    """

    transform_y_true = T.switch(T.eq(y_true, 0), -1, y_true)
    compl_y_pred = T.clip(T.sub(1., y_pred), 1e-20, 1)
    y_pred = T.clip(y_pred, 1e-20,1)
    logit = (T.log2(y_pred) - T.log2(compl_y_pred))
    return T.mean(T.maximum(1. - transform_y_true * logit, 0.), axis=-1)



def hingesig_tf(y_true, y_pred):
    """Computes the hingeles for a sigmoidal output by apply the logit to y_pred.
        Note: this function is intended for TENSORFLOW.

    Arguments:
        Arguments:
            y_true  -- a tensorflow tensor holding the true labels
            y_pred -- a tensorflow tensor holding the raw pradictions, i.e. the sigmoid output

    Returns:
        a tensorflow tensor with hingeloss
    """
    y_true = math_ops.to_float(y_true)

    all_ones = array_ops.ones_like(y_true)
    all_zeros = array_ops.zeros_like(y_true)
    y_true = math_ops.subtract(2 * y_true, all_ones)
    
    compl_y_pred = tf.clip_by_value(math_ops.sub(1., y_pred), 1e-20,1)
    y_pred = tf.clip_by_value(y_pred, 1e-20,1)
    logits = math_ops.log(math_ops.div(y_pred, compl_y_pred))
    logits_2 = math_ops.div(
        logits,
        math_ops.log(2.0),
    )
    return math_ops.mean(
        math_ops.maximum(
            math_ops.sub(1.0, math_ops.multiply(logits_2, y_true)),
            0.0
        ),
        axis=-1
    )
