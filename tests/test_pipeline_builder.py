# -*- coding: utf-8 -*-

import unittest
import logging
import numpy as np
import pandas as pd
from fresh.pipeline import PipeBuilder


logger = logging.getLogger(__name__)


class PipelineBuilderTestCase(unittest.TestCase):
    """
    Test specific expected functionality of the pipeline.PipeBuilder class
    """

    def test_dtype_inferences(self):
        """
        """
        series = pd.Series([1, 2, 'three', 4, 5])
        transformer = PipeBuilder._get_type_transformer(series)
        array = transformer.fit_transform(series.values.reshape(-1, 1))
        self.assertTrue(pd.isna(array[2]), 'Expected 2nd index to be NaN, got: "{}" instead.'.format(array[2]))
        logger.info('Dtype: {}'.format(array.dtype))

