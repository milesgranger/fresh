# -*- coding: utf-8 -*-

import os
import unittest
import logging
from pprint import pformat

import pandas as pd
import numpy as np

from sklearn import datasets, metrics
from sklearn.model_selection import cross_val_score

from fresh import Model


class DefaultModelTestCase(unittest.TestCase):
    """
    Test that the default model provided, given various data, can fulfill the 'train -> predict' workflow

    This test case we don't care so much about the optimization of the model, just that it is capable
    of looking at raw data and interpreting a model/pipeline which can cope with it.

    - NaN values are dealt with
    - Various target types (regression / classification (strings or numeric categories))
    - Determine text categories 'Blue, Red, ect' vs. text blobs 'This is some random text data'
    - Datetime values
    """

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):

        cls.data = dict()

        # Simple classification, features of numerical values, target of string categories
        cls.logger.debug('Loading Iris dataset.')
        iris = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'iris.csv'))
        cls.data['iris'] = {
            'X': iris[[c for c in iris.columns if c != 'species']],
            'y': iris['species']
        }

    def test_all_numeric_x_string_y_classification(self):
        """
        Numeric values and string targets
        """
        X = self.data['iris']['X']
        y = self.data['iris']['y']

        model = Model()
        model.fit(X, y)
        predictions = model.predict(X)
        self.logger.debug('Got predictions: {}'.format(pformat(predictions[:2])))

    def test_all_numeric_x_numeric_y_classification(self):
        """
        Numeric values and numeric categorical targets
        """
        X = self.data['iris']['X']
        y = self.data['iris']['y']
        y = y.astype('category').cat.codes

        model = Model()
        model.fit(X, y)
        predictions = model.predict(X)
        self.logger.debug('Got predictions: {}'.format(pformat(predictions[:2])))

    def test_all_numeric_x_regression(self):
        """
        Test basic regression
        """
        X, y = datasets.make_regression(n_samples=1000, n_features=50, n_informative=10)

        X = pd.DataFrame(X, columns=('x_{}'.format(i) for i in range(X.shape[1])))
        y = pd.Series(y)

        # Input random NaN values
        X = X.mask(np.random.random(X.shape) < 0.01)
        self.logger.debug('Total null values: {}'.format(pd.isna(X).sum().sum()))

        model = Model()
        model.fit(X, y)
        model.predict(X)
        scores = model.score(X, y)
        self.logger.debug('Regression avg score: {:.2f}, standard dev: {:.2f}'
                          .format(scores.mean(), scores.std()))
        self.assertGreaterEqual(scores.mean(), 0.5,
                                'Expected score of default model to be >= 0.5, got {:.2f}'.format(scores.mean()))
        self.assertLessEqual(scores.std(), 0.05,
                             msg='Expected stable model in cross validation, got STD of: {:.2f}'.format(scores.std()))



if __name__ == '__main__':
    unittest.main()
