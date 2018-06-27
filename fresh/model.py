# -*- coding: utf-8 -*-

import logging
from sklearn.base import BaseEstimator
from fresh.pipeline import PipeBuilder


class Model(BaseEstimator):

    target = None
    pipeline = None
    logger = logging.getLogger(__name__)

    def fit(self, X, y):
        if y.name in X.columns:
            self.logger.warning('Found target columns "{}" in X, deleting it!'.format(y.name))
            del X[y.name]
        self.pipeline = PipeBuilder.from_data(X, y)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
