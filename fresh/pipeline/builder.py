# -*- coding: utf-8 -*-


class PipeBuilder:
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


