import GPflowOpt
import GPflow
import numpy as np
import unittest
from GPflowOpt.models import MGP


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T


class TestMGP(unittest.TestCase):
    @property
    def domain(self):
        return np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])

    def create_parabola_model(self, design=None):
        if design is None:
            design = GPflowOpt.design.LatinHyperCube(16, self.domain)
        X, Y = design.generate(), parabola2d(design.generate())
        m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
        return m

    def test_object_integrity(self):
        m = self.create_parabola_model()
        Xs, Ys = m.X.value, m.Y.value
        n = MGP(m)

        self.assertEqual(n.wrapped, m)
        self.assertEqual(m._parent, n)
        self.assertTrue(np.allclose(Xs, n.X.value))
        self.assertTrue(np.allclose(Ys, n.Y.value))

    def test_predict_scaling(self):
        m = self.create_parabola_model()
        n = MGP(self.create_parabola_model())
        m.optimize()
        n.optimize()

        Xt = GPflowOpt.design.RandomDesign(20, self.domain).generate()
        fr, vr = m.predict_f(Xt)
        fs, vs = n.predict_f(Xt)
        self.assertTrue(np.shape(fr) == np.shape(fs))
        self.assertTrue(np.shape(vr) == np.shape(vs))
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))

        fr, vr = m.predict_y(Xt)
        fs, vs = n.predict_y(Xt)
        self.assertTrue(np.shape(fr) == np.shape(fs))
        self.assertTrue(np.shape(vr) == np.shape(vs))
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))

        fr, vr = m.predict_f_full_cov(Xt)
        fs, vs = n.predict_f_full_cov(Xt)
        self.assertTrue(np.shape(fr) == np.shape(fs))
        self.assertTrue(np.shape(vr) == np.shape(vs))
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))
