from keras.callbacks import Callback
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class _TestCallbackLog(Callback):
    """
        Testing purpose class.
    """
    def on_train_begin(self, logs=None):
        self.test_log = []

    def on_epoch_end(self, epoch, logs=None):
        self.test_log.append(epoch)


class GetAUPRCCallback(Callback):
    """Callback to comput the AUPRC and average precision socre at the end of each epoch.
    
    Arguments:
        Callback -- keras base class for all callbacks
    """
    def __init__(self, train_X, train_y, test_X, test_y):
        """constructor
        
        Arguments:
            train_X {np.ndarray} -- train data
            train_y {np.ndarray} -- train labels
            test_X {np.ndarray} -- test data
            test_y {np.ndarray} -- test labels
        """
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.test_AUPRC = None
        self.train_AUPRC = None
        self.test_AVG_PREC = None
        self.train_AVG_PREC = None

    def on_train_begin(self, logs={}):
        """ Init of all lists of scores"""
        self.test_AUPRC = []
        self.train_AUPRC = []
        self.test_AVG_PREC = []
        self.train_AVG_PREC = []

    def on_epoch_end(self, epoch, logs=None):
        """" On epoch end we compute the AUPRC and the average precision score. """
        test_probas = self.model.predict(self.test_X)
        train_probas = self.model.predict(self.train_X)

        p, r, _ = precision_recall_curve(y_true=self.test_y, probas_pred=test_probas)
        self.test_AUPRC.append(auc(x=r,y=p))

        p, r, _ = precision_recall_curve(y_true=self.train_y, probas_pred=train_probas)
        self.train_AUPRC.append(auc(x=r,y=p))


        score = average_precision_score(y_true=self.test_y, y_score=test_probas)
        self.test_AVG_PREC.append(score)

        score = average_precision_score(y_true=self.train_y, y_score=train_probas)
        self.train_AVG_PREC.append(score)
