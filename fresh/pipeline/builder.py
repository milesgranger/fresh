# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, Imputer

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
            ('feature_{}'.format(feature), cls._make_feature_pipeline_transformer(X[feature], feature))
            for feature in X.columns
        ]
        return FeatureUnion(transformer_list=transformer_list,
                            n_jobs=1)

    @classmethod
    def _make_feature_pipeline_transformer(cls, series: pd.Series, feature: str) -> Pipeline:
        """
        Given a series, determine the most appropriate transformer and hand that back

        ie. if blobs of text are found, return a pipeline of HashVectorizers into TFIDF transformers.
            or if biases numerical distribution is found, return Log1p transformer
        """
        pipe = Pipeline(steps=[
            ('{}_selector'.format(feature), Selector(feature)),
            ('{}_dtype_conversion', cls._get_type_transformer(series)),
            ('{}_nan_imputer', Imputer(strategy='median'))
        ])
        return pipe


    @staticmethod
    def _get_type_transformer(series: pd.Series) -> FunctionTransformer:
        """
        Determine the predominant type of a series and return a sklearn transformer to convert new series to
        that datatype. ie. if series contains [1, 2, 'NA', 3, 4] it will replace 'NA' with np.NaN
        """
        def attempt_conversion(array):

            array = array.flatten() if hasattr(array, 'flatten') else array.values.flatten()

            for i, val in enumerate(array):
                try:
                    array[i] = cast_func(val)
                except ValueError:
                    array[i] = np.NaN
            return array.reshape(-1, 1)

        def determine_type(val):
            try:
                return type(literal_eval(val))
            except ValueError:
                return type(val)

        cast_func = series.map(determine_type).value_counts().index[0]

        return FunctionTransformer(func=attempt_conversion, validate=False)

