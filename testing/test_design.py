import GPflowOpt
import unittest
import numpy as np


class _DesignTests(unittest.TestCase):
    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -i, 2 * i) for i in range(1, 4)])

    def test_design_compliance(self):
        points = self.design.generate()
        self.assertTupleEqual(points.shape, (self.design.size, self.design.domain.size), msg="Generated design does not"
                                                                                             "satisfy specifications")
        self.assertIn(points, self.design.domain, "Not all generated points are generated within the domain")


class TestRandomDesign(_DesignTests):
    def setUp(self):
        super(TestRandomDesign, self).setUp()
        self.design = GPflowOpt.design.RandomDesign(200, self.domain)


class TestEmptyDesign(_DesignTests):
    def setUp(self):
        super(TestEmptyDesign, self).setUp()
        self.design = GPflowOpt.design.EmptyDesign(self.domain)


class TestFactorialDesign(_DesignTests):
    def setUp(self):
        super(TestFactorialDesign, self).setUp()
        self.design = GPflowOpt.design.FactorialDesign(4, self.domain)

    def test_validity(self):
        A = self.design.generate()
        for i in range(1, self.domain.size + 1):
            self.assertTrue(np.all(np.any(A[i - 1, :] - np.linspace(-i, 2 * i, 4)[:, None] < 1e-4, axis=0)),
                            msg="Generated off-grid.")
