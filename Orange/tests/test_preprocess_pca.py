import unittest

from Orange.data import Table
from Orange.preprocess import ProjectPCA


class TestPCAProjector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ionosphere = Table("ionosphere")

    def project_pca_helper(self, projector, res):
        data = self.ionosphere
        data_pc = projector(data)
        self.assertEqual(data_pc.X.shape[1], res)
        self.assertTrue((data.metas == data_pc.metas).all())
        self.assertTrue((data.Y == data_pc.Y).any())

    def test_project_pca_default(self):
        self.project_pca_helper(ProjectPCA(), self.ionosphere.X.shape[1])

    def test_project_pca(self):
        self.project_pca_helper(ProjectPCA(n_components=5), 5)