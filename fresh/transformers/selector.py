# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.base import TransformerMixin


class Selector(TransformerMixin):
    """
    Select a column from a pandas dataframe
    """
    def __init__(self, feature: str):
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[[self.feature]].values.reshape(-1, 1)

    def __repr__(self):
        return 'fresh.transformers.Selector("{}")'.format(self.feature)
