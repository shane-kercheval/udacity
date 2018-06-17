import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def column_log(x):
    return np.log(x + 1)


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
