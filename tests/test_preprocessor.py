# -*- coding: utf-8 -*-

import unittest


class PreprocessorTestCase(unittest.TestCase):
    """
    Test that the pre-processor can successfully execute desired functionality

    - Convert data types
    - Convert targets from string to numerical values in the case of categorical encodings
    - Deal with NaNs
    """
    def setUp(self):
        pass

    def test_nan_dealings(self):
        """
        Ensure datasets with NaNs can be dealt with.
        """


if __name__ == '__main__':
    unittest.main()
