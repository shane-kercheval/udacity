from typing import Union

import numpy as np
import oolearning as oo
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


def column_log(x):
    return np.log(x + 1)


def create_net_capital(x):
    temp = x.copy()
    temp['net capital'] = temp['capital-gain'] - temp['capital-loss']
    return temp


def strip_strings(x):
    return x.str.strip()


# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if self.attribute_names is None else X[self.attribute_names].values


class AppendFeatures(BaseEstimator, TransformerMixin):
    """
    simply returns the dataset, to be used in a FeatureUnion
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        return self.dataset


class CustomLogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, base=None):
        """
        :param base: base of the log; None is no transform
        """
        self.base = base

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.base is None:
            return X
        # LOGARITHM BASE CHANGE RULE: log_b(x) = log_c(x) / log_c(b)
        # np.log is base e

        return np.log(X + 1) / np.log(self.base)


class ChooserTransform(BaseEstimator, TransformerMixin):
    def __init__(self, base_transformer=None):
        self.base_transformer = base_transformer

    def fit(self, X, y=None):
        if self.base_transformer is None:
            return self

        return self.base_transformer.fit(X, y)

    def transform(self, X):
        if self.base_transformer is None:
            return X

        return self.base_transformer.transform(X)


class CombineAgeHoursTransform(BaseEstimator, TransformerMixin):
    def __init__(self, combine=True):
        """
        :param combine: used to hyper-tune
        """
        self.combine = combine

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.combine is False:
            return X

        else:
            age_X = X[:, 0]
            education_X = X[:, 1]
            hours_X = X[:, 2]

            return np.c_[age_X, education_X, hours_X, age_X * hours_X]


class CombineCapitalGainLossTransform(BaseEstimator, TransformerMixin):
    def __init__(self, combine=True):
        """
        :param combine: used to hyper-tune
        """
        self.combine = combine

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.combine is False:
            return X

        else:
            capital_gain = X[:, 0]
            capital_loss = X[:, 1]

            return np.c_[capital_gain, capital_loss, capital_gain - capital_loss]


class BinaryAucRocScore(oo.AucRocScore):
    """
    Calculates the AUC of the ROC curve as defined by sklearn's `roc_auc_score()`
        http://scikit-learn.org/stable/modules/generated/sklearn.score_names.roc_auc_score.html
    """
    def __init__(self, positive_class, threshold: float=0.5):
        super().__init__(positive_class=positive_class)
        self._threshold = threshold

    @property
    def name(self) -> str:
        return 'BINARY_AUC'

    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:

        return roc_auc_score(y_true=[1 if x == self._positive_class else 0 for x in actual_values],
                             # binary makes it so it converts the "scores" to predictions
                             y_score=[1 if x > self._threshold else 0 for x in predicted_values[self._positive_class].values])
