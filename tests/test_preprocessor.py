# -*- coding: utf-8 -*-

import os
import unittest
import logging
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
        model = Model(target='species')
        self.logger.debug('Start of fitting model.')
        model.fit(self.iris)


if __name__ == '__main__':
    unittest.main()
