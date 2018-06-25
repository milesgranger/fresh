# -*- coding: utf-8 -*-


class Model:

    target = None

    def __init__(self, target: str):
        self.target = target

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

