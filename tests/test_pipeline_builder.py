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
        Infer the primary dtype for a given array.

        Occasionally, an array might be floats, but NaNs represented as 'N/A' or some other string/value,
        the builder should find these and replace them with NaN values OR cast them at the primary dtype in that array
        """

        # Each 2nd index value should be returned as NaN or as the type of the other values
        for vec_to_infer in [
            [1, 2, 'three', 4, 5],
            ['one', 'two', 3, 'four', 'five'],
            [1.0, 2.0, 3, 4.0, 5.0],
            [1, 'Null', 2, 'None', 4, 5, None],
        ]:
            series = pd.Series(vec_to_infer)
            transformer = PipeBuilder._get_type_transformer(series)
            array = transformer.fit_transform(series.values.reshape(-1, 1))

            # The 2nd index is either NaN or the same instance type of the first element.
            self.assertTrue(pd.isna(array[2]) or isinstance(array[2][0], type(array[0][0])),
                            'Expected 2nd index to be NaN, got: "{}" instead.'.format(array[2]))

    def test_regression_or_classification(self):
        """
        Test various scenarios of a classification or regression based target.
        """
        for problem_type, array in [
            ('regression', np.arange(start=0, stop=100)),
            ('regression', np.random.random_sample(size=100)),
            ('regression', np.random.randint(low=0, high=100, size=100)),
            ('classification', np.random.randint(low=0, high=3, size=100)),
            ('classification', np.random.randint(low=5, high=10, size=100)),
            ('classification', np.asarray(['A', 'B', 'C', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'A', 'C', 'B']))
        ]:
            assumed_problem_type = PipeBuilder._determine_classification_or_regression(array)
            msg = (
                'Expecting problem type: "{}", got "{}" from array: {}'.format(
                    problem_type, assumed_problem_type, array[:5])
            )
            logger.debug(msg)
            self.assertEqual(problem_type, assumed_problem_type, msg=msg)
