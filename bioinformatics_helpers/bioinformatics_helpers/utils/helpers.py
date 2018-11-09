import numpy as np
from sklearn.metrics import precision_recall_curve
import copy



def get_iqr_outliers(data):
    """
    outliers determined by the IQR rule
    """
    _data = np.array(data)
    q1_q3 = np.percentile(_data, [25,75])
    iqr = q1_q3[1] - q1_q3[0]
    low_value = q1_q3[0] - (1.5 * iqr)
    high_value = q1_q3[1] + (1.5 * iqr)
    result = {
        "low_value" : low_value,
        "high_value" : high_value,
        "high_outliers" : _data[ data > high_value],
        "low_outliers" : _data[data < low_value]
    }
    return result
    #high outliers in negative class


def interpolated_precision_recall_curve(y_true, y_prob):
    """
        returns: tuple (x,y) where x=recallInverse, y=decreasing_precision. 
        See https://stackoverflow.com/questions/39836953/how-to-draw-a-precision-recall-curve-with-interpolation-in-python
        for more information.
    """
    p, r, _ = precision_recall_curve(y_true=y_true,probas_pred=y_prob)
    pr = copy.deepcopy(p)
    rec = copy.deepcopy(r)
    prInv = np.fliplr([pr])[0]
    recInv = np.fliplr([rec])[0]
    j = rec.shape[0]-2
    while j>=0:
        if prInv[j+1]>prInv[j]:
            prInv[j]=prInv[j+1]
        j=j-1
    decreasing_max_precision = np.maximum.accumulate(prInv[::-1])[::-1]
    return(recInv, decreasing_max_precision)



