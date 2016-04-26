import unittest
import os

import numpy as np

from Orange.data import io, ContinuousVariable, DiscreteVariable, StringVariable


def read_file(name):
    return io.ExcelFormat().read_file(
        os.path.join(os.path.dirname(__file__), "xlsx_files", name))


def meta_test_helper(domain, table):
    test = unittest.TestCase('__init__')
    test.assertEqual(len(domain.metas), 3)
    for n, var in zip("acf", domain.metas):
        test.assertEqual(var.name, n)
    test.assertIsInstance(domain.metas[0], DiscreteVariable)
    test.assertEqual(domain.metas[0].values, ["green", "red"])
    test.assertIsInstance(domain.metas[1], ContinuousVariable)
    np.testing.assert_almost_equal(table.metas[:, 0], np.array([1, 1, 0] * 7 + [1, 1]))
    np.testing.assert_almost_equal(table.metas[:, 1], np.array([0, 1, 2, 3] * 5 + [0, 1, 2]))
    np.testing.assert_equal(table.metas[:, 2], np.array(list("abcdefghijklmnopqrstuvw")))

class TestExcelHeader0(unittest.TestCase):
    def test_read(self):
        table = read_file("header_0.xlsx")
        domain = table.domain
        self.assertIsNone(domain.class_var)
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 4)
        for i, var in enumerate(domain.attributes):
            self.assertIsInstance(var, ContinuousVariable)
            self.assertEqual(var.name, "Feature {}".format(i + 1))
        np.testing.assert_almost_equal(table.X,
                                       np.array([[0.1, 0.5, 0.1, 21],
                                                 [0.2, 0.1, 2.5, 123],
                                                 [0, 0, 0, 0]]))


class TextExcelSheets(unittest.TestCase):
    def test_named_sheet(self):
        table = read_file("header_0_sheet.xlsx:my_sheet")
        self.assertEqual(len(table.domain.attributes), 4)


class TestExcelHeader1(unittest.TestCase):
    def test_no_flags(self):
        table = read_file("header_1_no_flags.xlsx")
        domain = table.domain
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 4)
        self.assertIsInstance(domain[0], DiscreteVariable)
        self.assertIsInstance(domain[1], ContinuousVariable)
        self.assertIsInstance(domain[2], DiscreteVariable)
        self.assertIsInstance(domain[3], ContinuousVariable)
        for i, var in enumerate(domain):
            self.assertEqual(var.name, chr(97 + i))
        self.assertEqual(domain[0].values, ["green", "red"])
        np.testing.assert_almost_equal(table.X,
                                       np.array([[1, 0.5, 0, 21],
                                                 [1, 0.1, 0, 123],
                                                 [0, 0, np.nan, 0]]))
        np.testing.assert_equal(table.Y, np.array([]).reshape(3, 0))

    def test_flags(self):
        table = read_file("header_1_flags.xlsx")
        domain = table.domain

        np.testing.assert_almost_equal(table.X, np.arange(23).reshape(23, 1))
        np.testing.assert_almost_equal(table.Y, np.array([.5, .1, 0, 0] * 5 + [.5, .1, 0]))

        self.assertEqual(len(domain.attributes), 1)
        self.assertEqual(len(domain.class_vars), 1)
        for value, realName in ((domain.attributes[0], "d"),
                                (domain.class_var, "b")):
            self.assertEqual(value.name, realName)
            self.assertIsInstance(value, ContinuousVariable)
        meta_test_helper(domain, table)


class TestExcelHeader3(unittest.TestCase):
    def test_read(self):
        table = read_file("header_3.xlsx")
        domain = table.domain

        np.testing.assert_almost_equal(table.X[:, 0], np.arange(23))
        np.testing.assert_almost_equal(table.X[:, 1], np.array([1, 0] + [float("nan")] * 21))
        np.testing.assert_almost_equal(table.Y, np.array([.5, .1, 0, 0] * 5 + [.5, .1, 0]))

        self.assertEqual(len(domain.attributes), 2)
        self.assertEqual(len(domain.class_vars), 1)

        for vals, realName, valueType in ((domain.attributes[0], "d", ContinuousVariable),
                                          (domain.attributes[1], "g", DiscreteVariable),
                                          (domain.class_var, "b", ContinuousVariable)):
            self.assertEqual(vals.name, realName)
            self.assertIsInstance(vals, valueType)
        meta_test_helper(domain, table)

if __name__ == "__main__":
    unittest.main()
