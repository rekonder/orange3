from Orange.data import Table
from Orange.widgets.classify.owknn import OWKNNLearner
from Orange.widgets.tests.base import GuiTest

class WidgetTests(GuiTest):
    def setUp(self):
        self.widget = OWKNNLearner()

    def test_visible_spinner(self):
        inputs = self.widget.button_opts
        for i in range(0, 3):
            self.assertEqual(not inputs[i].isHidden(), True)

    def test_neighbours(self):
        pass

    def test_metric(self):
        pass

    def test_weight(self):
        pass
