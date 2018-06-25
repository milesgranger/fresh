# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from fresh.pipeline import PipeBuilder


class Model(BaseEstimator):

    target = None
    pipeline = None

    def fit(self, X, y):
        self.pipeline = PipeBuilder.from_data(X, y)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
