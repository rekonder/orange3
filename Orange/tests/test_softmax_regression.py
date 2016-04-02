import unittest

from Orange.data import ContinuousVariable, Domain, Table
from Orange.classification import Model, SoftmaxRegressionLearner
from Orange.evaluation import CrossValidation, CA
import numpy as np


class SoftmaxRegressionTest(unittest.TestCase):
    def test_SoftmaxRegression(self):
        table = Table('iris')
        learner = SoftmaxRegressionLearner()
        results = CrossValidation(table, [learner], k=3)
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 1.0)

    def test_SoftmaxRegressionPreprocessors(self):
        table = Table('iris')
        table.X[:,2] = table.X[:,2] * 0.001
        table.X[:,3] = table.X[:,3] * 0.001
        learners = [SoftmaxRegressionLearner(preprocessors=[]),
                    SoftmaxRegressionLearner()]
        results = CrossValidation(table, learners, k=10)
        ca = CA(results)
        self.assertTrue(ca[0] < ca[1])

    def test_probability(self):
        table = Table('iris')
        learn = SoftmaxRegressionLearner()
        clf = learn(table)
        p = clf(table, ret=Model.Probs)
        self.assertLess(abs(p.sum(axis=1) - 1).all(), 1e-6)

    def test_predict_table(self):
        table = Table('iris')
        learner = SoftmaxRegressionLearner()
        c = learner(table)
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Table('iris')
        learner = SoftmaxRegressionLearner()
        c = learner(table)
        c(table.X)
        vals, probs = c(table.X, c.ValueProbs)
