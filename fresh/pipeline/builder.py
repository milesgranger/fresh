# -*- coding: utf-8 -*-

import logging
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from fresh.transformers import Selector


logger = logging.getLogger(__name__)


class PipeBuilder(Pipeline):
    """
    Analyze dataset and return a sensible sklearn Pipeline

    Structure of pipeline:
        FeatureUnion -> join together modified/transformed features of dataset
        PCA -> Tune-able parameter of pipeline in grid search
        Model -> Classification or Regression based model.

    Extra attributes:
        ._problem_type = "regression" or "classification"  # determine which models are suitable in grid search
        ._n_features   = int # Number of features in raw dataset.

    Usage:
    >>> pipeline = PipeBuilder.from_data(X, y)  # Where X & y are of type pandas.core.DataFrame
    """

    @classmethod
    def from_data(cls, X, y):
        """
        Return a scikit-learn pipeline from raw dataset.
        """
        # Base pipeline which should be ran through a parallelized gridsearch to swap out models, and do other
        # modifications to find the best pipeline.
        steps = [
            ('features', cls._build_feature_union_step(X)),
            ('pca', PCA(n_components=X.shape[1])),
            ('model', RandomForestClassifier(min_samples_split=25))
        ]
        return cls(steps=steps)


    @classmethod
    def _build_feature_union_step(cls, X: pd.DataFrame) -> FeatureUnion:
        """
        Given a dataframe of features, return a FeatureUnion which transforms each feature accordingly and joins them
        """
        transformer_list = [
            ('feature_{}'.format(feature), cls._get_best_transformer(X[feature], feature))
            for feature in X.columns
        ]
        return FeatureUnion(transformer_list=transformer_list,
                            n_jobs=1)


    @staticmethod
    def _get_best_transformer(series: pd.Series, feature: str) -> Pipeline:
        """
        Given a series, determine the most appropriate transformer and hand that back

        ie. if blobs of text are found, return a pipeline of HashVectorizers into TFIDF transformers.
            or if biases numerical distribution is found, return Log1p transformer
        """
        pipe = Pipeline(steps=[
            ('{}_selector'.format(feature), Selector(feature)),
        ])
        return pipe

