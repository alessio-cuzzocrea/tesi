import theano.tensor as T
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import numpy as np

def hingesig(y_true, y_pred):
    transform_y_true = T.switch(T.eq(y_true, 0), -1, y_true)
    compl_y_pred = T.clip(T.sub(1., y_pred), 1e-20, 1)
    y_pred = T.clip(y_pred, 1e-20,1)
    logit = (T.log2(y_pred) - T.log2(compl_y_pred))
    return T.mean(T.maximum(1. - transform_y_true * logit, 0.), axis=-1)

def logit_theano(y_true, y_pred):
    transform_y_true = T.switch(T.eq(y_true,0), -1, y_true)
    clipped_y_pred = T.clip(y_pred, 1e-20, 0.9999999)
    logit = (T.log2(clipped_y_pred) - T.log2(T.sub(1., clipped_y_pred)))
    return logit


def hingesig_tf(y_true, y_pred):
    ###logits = math_ops.log2(y_pred) - math_ops.log2(math_ops.sub(1., y_pred))
    y_true = math_ops.to_float(y_true)
    #y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
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
