import unittest
import numpy as np
from Orange.data import _valuecount


class test_valuecount(unittest.TestCase):

    def valuecount_helper(self, *args):
        for arg in args:
            a = np.array(arg[0])
            b = _valuecount.valuecount(a)
            if len(arg) == 2:
                np.testing.assert_almost_equal(b, arg[-1])
            else:
                np.testing.assert_almost_equal(b, a)

    def valuecount_raises_helper(self, *args):
        for arg in args:
            self.assertRaises(TypeError, _valuecount.valuecount, arg)

    def test_valuecount(self):
        self.valuecount_helper([[[1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]], [[1], [1]]],
                               [[[1, 1, 1, 2], [0.1, 0.2, 0.3, 0.4]], [[1, 2], [0.6, 0.4]]],
                               [[[0, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]], [[0, 1], [0.1, 0.9]]],
                               [[[0, 1, 1, 2], [0.1, 0.2, 0.3, 0.4]], [[0, 1, 2], [0.1, 0.5, 0.4]]],
                               [[[0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]]],
                               [[[0], [0.1]]],
                               [np.ones((2, 1))])

        self.valuecount_raises_helper([np.array([[0, 1], [2, 3]])],
                                      [np.ones(2)],
                                      [np.ones((3, 3))],
                                      [None])