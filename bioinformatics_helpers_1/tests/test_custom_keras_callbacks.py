import unittest
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from bioinformatics_helpers.utils import _TestCallbackLog
from bioinformatics_helpers.utils import GetAUPRCCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from unittest.mock import patch, Mock
from unittest import mock
import keras


class TestCustomKerasCallbacks(unittest.TestCase):

    def get_model(self):
        model = Sequential([
            Dense(32, input_shape=(1,)),
            Activation('relu'),
            Dense(1),
            Activation('softmax'),
        ])
        return model
    
    def test_log(self):
        X = [1,2,3,4,5,6]
        y = [1,1,1,0,0,0]
        callback = _TestCallbackLog()
        model = self.get_model()
        model.compile(optimizer='rmsprop',
                      loss='mse',
                      )
        history = model.fit(X,y,epochs=5, callbacks=[callback], verbose=False)
        self.assertListEqual(callback.test_log,  [0,1,2,3,4])
    
    def test_get_AUPRCCallback(self):
        train_X = [-1,-2,-3,4,5,6]
        train_y = [1,1,1,0,0,0]
        test_X = [7,8,9,-10,-1,0]
        test_y = [0,0,0,1,1,0]

        def fake_predict(X):
            return np.random.uniform(0,1,len(X))

        with mock.patch('keras.models.Model', spec=keras.models.Model) as mock_model:
            # todo: return value da sostiture con funzione stub
            mock_model.predict.side_effect = fake_predict
            callback = GetAUPRCCallback(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
            callback.set_model(mock_model)
            callback.on_train_begin()
            for epoch in range(0, 5):
                np.random.seed(epoch)
                callback.on_epoch_end(epoch)
                np.random.seed(epoch)
                test_probas = fake_predict(test_X)
                train_probas = fake_predict(train_X)
                p, r, _ = precision_recall_curve(y_true=test_y, probas_pred=test_probas)
                self.assertEqual(auc(x=r, y=p), callback.test_AUPRC[-1])
                p, r, _ = precision_recall_curve(y_true=train_y, probas_pred=train_probas)
                self.assertEqual(auc(x=r, y=p), callback.train_AUPRC[-1])

    def test_get_AUPRCCallback_keras(self):
        train_X = [-1,-2,-3,4,5,6]
        train_y = [1,1,1,0,0,0]
        test_X = [7,8,9,-10,-1,0]
        test_y = [0,0,0,1,1,0]
        callback = GetAUPRCCallback(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
        model = self.get_model()
        model.compile(optimizer='rmsprop',
                      loss='mse',
                      )
        model.fit(train_X, train_X, epochs=5, callbacks=[callback], verbose=False)
        probas = model.predict(test_X)
        p, r, _ = precision_recall_curve(y_true=test_y, probas_pred=probas)
        self.assertEqual(auc(x=r,y=p), callback.test_AUPRC[-1])

        probas = model.predict(train_X)
        p, r, _ = precision_recall_curve(y_true=train_y, probas_pred=probas)
        self.assertEqual(auc(x=r, y=p), callback.train_AUPRC[-1])


if __name__ == '__main__':
    unittest.main()
        