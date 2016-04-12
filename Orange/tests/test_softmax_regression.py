import unittest

from Orange.data import Table
from Orange.classification import Model, SoftmaxRegressionLearner
from Orange.evaluation import CrossValidation, CA


class SoftmaxRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.learner = SoftmaxRegressionLearner()
        cls.clf = cls.learner(cls.iris)

    def test_SoftmaxRegression(self):
        results = CrossValidation(self.iris, [self.learner], k=3)
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 1.0)

    def test_SoftmaxRegressionPreprocessors(self):
        table = self.iris.copy()
        table.X[:, 2] = table.X[:, 2] * 0.001
        table.X[:, 3] = table.X[:, 3] * 0.001
        learners = [SoftmaxRegressionLearner(preprocessors=[]),
                    self.learner]
        results = CrossValidation(table, learners, k=10)
        ca = CA(results)
        self.assertLess(ca[0], ca[1])

    def test_probability(self):
        p = self.clf(self.iris, ret=Model.Probs)
        self.assertLess(abs(p.sum(axis=1) - 1).all(), 1e-6)

    def test_predict_table(self):
        self.clf(self.iris)
        vals, probs = self.clf(self.iris, self.clf.ValueProbs)

    def test_predict_numpy(self):
        self.clf(self.iris.X)
        vals, probs = self.clf(self.iris.X, self.clf.ValueProbs)
