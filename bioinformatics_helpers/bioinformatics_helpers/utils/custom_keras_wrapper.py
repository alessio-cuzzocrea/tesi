"""Wrapper for using the Scikit-Learn API with Keras models.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import types
import inspect
import numpy as np

from keras.models import Sequential


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def has_arg(fn, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.
    For Python 2, checks if there is an argument with the given name.
    For Python 3, checks if there is an argument with the given name, and
    also whether this argument can be called with a keyword (i.e. if it is
    not a positional-only argument).
    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name`
                    but the function accepts a `**kwargs` argument.
    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    """
    if sys.version_info < (3,):
        arg_spec = inspect.getargspec(fn)
        if accept_all and arg_spec.keywords is not None:
            return True
        return (name in arg_spec.args)
    elif sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(fn)
        if accept_all and arg_spec.varkw is not None:
            return True
        return (name in arg_spec.args or
                name in arg_spec.kwonlyargs)
    else:
        signature = inspect.signature(fn)
        parameter = signature.parameters.get(name)
        if parameter is None:
            if accept_all:
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        return True
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))


class BaseWrapper(object):
    """Base class for the Keras scikit-learn wrapper.
    Warning: This class should not be used directly.
    Use descendant classes instead.
    # Arguments
        build_fn: callable function or class instance
        **sk_params: model parameters & fitting parameters
    The `build_fn` should construct, compile and return a Keras model, which
    will then be used to fit/predict. One of the following
    three values could be passed to `build_fn`:
    1. A function
    2. An instance of a class that implements the `__call__` method
    3. None. This means you implement a class that inherits from either
    `KerasClassifier` or `KerasRegressor`. The `__call__` method of the
    present class will then be treated as the default `build_fn`.
    `sk_params` takes both model parameters and fitting parameters. Legal model
    parameters are the arguments of `build_fn`. Note that like all other
    estimators in scikit-learn, `build_fn` should provide default values for
    its arguments, so that you could create the estimator without passing any
    values to `sk_params`.
    `sk_params` could also accept parameters for calling `fit`, `predict`,
    `predict_proba`, and `score` methods (e.g., `epochs`, `batch_size`).
    fitting (predicting) parameters are selected in the following order:
    1. Values passed to the dictionary arguments of
    `fit`, `predict`, `predict_proba`, and `score` methods
    2. Values passed to `sk_params`
    3. The default values of the `keras.models.Sequential`
    `fit`, `predict`, `predict_proba` and `score` methods
    When using scikit-learn's `grid_search` API, legal tunable parameters are
    those you could pass to `sk_params`, including fitting parameters.
    In other words, you could use `grid_search` to search for the best
    `batch_size` or `epochs` as well as the model parameters.
    """

    def __init__(self, build_fn=None, generator=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.generator = generator
        self.check_params(sk_params)

        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

    def check_params(self, params):
        """Checks for user typos in `params`.
        # Arguments
            params: dictionary; the parameters to be checked
        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        legal_params_fns = [Sequential.fit, Sequential.predict,
                            Sequential.predict_classes, Sequential.evaluate, Sequential.fit_generator]
        if self.build_fn is None:
            legal_params_fns.append(self.__call__)
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            legal_params_fns.append(self.build_fn.__call__)
        else:
            legal_params_fns.append(self.build_fn)
        # append generator params
        if self.generator is not None:
            legal_params_fns.append(self.generator.__init__)

        for params_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, params_name):
                    break
            else:
                if params_name != 'nb_epoch':
                    raise ValueError(
                        '{} is not a legal parameter'.format(params_name))

    def get_params(self, **params):
        """Gets parameters for this estimator.
        # Arguments
            **params: ignored (exists for API compatibility).
        # Returns
            Dictionary of parameter names mapped to their values.
        """
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        res.update({"generator": self.generator})
        return res

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        # Arguments
            **params: Dictionary of parameter names mapped to their values.
        # Returns
            self
        """
        self.check_params(params)
        self.sk_params.update(params)
        return self

    def fit(self, x, y, **kwargs):
        """ fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        """


        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        if self.generator is not None:
            gen_params = self.filter_gen_params()
            gen = self.generator(x,y, **gen_params)
            fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
            fit_args.update(kwargs)
            history = self.model.fit_generator(gen, **fit_args)
        else:
            fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
            fit_args.update(kwargs)
            history = self.model.fit(x, y, **fit_args)

        return history

    def filter_sk_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.
        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`
        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res

    def filter_gen_params(self, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.
        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`
        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(self.generator.__init__, name):
                res.update({name: value})
        res.update(override)
        return res


class CustomKerasClassifier(BaseWrapper):
    """Implementation of the scikit-learn classifier API for Keras.
    """

    def fit(self, x, y, sample_weight=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        # Raises
            ValueError: In case of invalid shape for `y` argument.
        """
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        return super(CustomKerasClassifier, self).fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.
        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)

        proba = self.model.predict(x, **kwargs)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]

    def predict_proba(self, x, **kwargs):
        """Returns class probability estimates for the given test data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.
        # Returns
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                to match the scikit-learn API,
                will return an array of shape `(n_samples, 2)`
                (instead of `(n_sample, 1)` as in Keras).
        """
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        probs = self.model.predict(x, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.
        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.
        # Raises
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
        """
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')