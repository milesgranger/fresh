# -*- coding: utf-8 -*-

import os
import unittest
import logging
from pprint import pformat
import pandas as pd

from fresh import Model


class PipelineBuilderTestCase(unittest.TestCase):
    """
    Test that the pipeline builder can successfully execute desired functionality

    - Convert data types
    - Convert targets from string to numerical values in the case of categorical encodings
    - Deal with NaNs
    """

    logger = logging.getLogger(__name__)

    def setUp(self):
        self.logger.debug('Loading Iris dataset.')
        self.iris = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'iris.csv'))

    def test_basic_classification(self):
        """
        Ensure datasets with NaNs can be dealt with.
        """
        model = Model()
        self.logger.debug('Start of fitting model.')
        model.fit(X=self.iris[[c for c in self.iris.columns if c != 'species']],
                  y=self.iris['species'])
        predictions = model.predict(self.iris[[c for c in self.iris.columns if c != 'species']])
        self.logger.info('Got predictions: {}'.format(pformat(predictions[:10])))



if __name__ == '__main__':
    unittest.main()
