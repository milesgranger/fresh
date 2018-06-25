# -*- coding: utf-8 -*-

from sklearn.base import TransformerMixin


class Selector(TransformerMixin):
    """
    Select a column from a pandas dataframe
    """
    def __init__(self, feature: str):
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.feature]]

    def __repr__(self):
        return 'fresh.transformers.Selector("{}")'.format(self.feature)
