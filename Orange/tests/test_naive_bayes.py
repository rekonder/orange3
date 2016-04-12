import unittest

import Orange
from Orange.classification import NaiveBayesLearner
from Orange.data import Table
from Orange.evaluation import CrossValidation,CA

class NaiveBayesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = Table('titanic')
        cls.learner = NaiveBayesLearner()
        cls.learned = cls.learner(data)
        cls.table = data[::20]

    def test_NaiveBayes(self):
        results = CrossValidation(self.table, [self.learner], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

    def test_predict_single_instance(self):
        for ins in self.table:
            self.learned(ins)
            val, prob = self.learned(ins, self.learned.ValueProbs)

    def test_predict_table(self):
        self.learned(self.table)
        vals, probs = self.learned(self.table, self.learned.ValueProbs)

    def test_predict_numpy(self):
        X = self.table.X[::20]
        self.learned(X)
        vals, probs = self.learned(X, self.learned.ValueProbs)
