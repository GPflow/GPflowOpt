import GPflowOpt
import GPflow
import numpy as np
import unittest
from GPflowOpt.scaling import DataScaler


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T


class TestDataScaler(unittest.TestCase):
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
        n = DataScaler(m, self.domain)

        self.assertEqual(n.wrapped, m)
        self.assertEqual(m._parent, n)
        self.assertTrue(np.allclose(Xs, n.X.value))
        self.assertTrue(np.allclose(Ys, n.Y.value))

    def test_enabling_transforms(self):
        m = self.create_parabola_model()
        normY = (m.Y.value - np.mean(m.Y.value, axis=0)) / np.std(m.Y.value, axis=0)
        scaledX = (m.X.value + 1) / 2

        n1 = DataScaler(m, normalize_Y=False)
        self.assertFalse(n1.normalize_output)
        self.assertTrue(np.allclose(m.X.value, n1.X.value))
        self.assertTrue(np.allclose(m.Y.value, n1.Y.value))
        n1.input_transform = self.domain >> GPflowOpt.domain.UnitCube(self.domain.size)
        self.assertTrue(np.allclose(m.X.value, scaledX))
        self.assertTrue(np.allclose(m.Y.value, n1.Y.value))
        n1.normalize_output = True
        self.assertTrue(n1.normalize_output)
        self.assertTrue(np.allclose(m.Y.value, normY))

        m = self.create_parabola_model()
        n2 = DataScaler(m, self.domain, normalize_Y=False)
        self.assertTrue(np.allclose(m.X.value, scaledX))
        self.assertTrue(np.allclose(m.Y.value, n2.Y.value))
        n2.normalize_output = True
        self.assertTrue(np.allclose(m.Y.value, normY))
        n2.input_transform = GPflowOpt.domain.UnitCube(self.domain.size) >> GPflowOpt.domain.UnitCube(self.domain.size)
        self.assertTrue(np.allclose(m.X.value, n1.X.value))

        m = self.create_parabola_model()
        n3 = DataScaler(m, normalize_Y=True)
        self.assertTrue(np.allclose(m.X.value, n3.X.value))
        self.assertTrue(np.allclose(m.Y.value, normY))
        n3.normalize_output = False
        self.assertTrue(np.allclose(m.Y.value, n3.Y.value))

        m = self.create_parabola_model()
        n4 = DataScaler(m, self.domain, normalize_Y=True)
        self.assertTrue(np.allclose(m.X.value, scaledX))
        self.assertTrue(np.allclose(m.Y.value, normY))
        n4.normalize_output = False
        self.assertTrue(np.allclose(m.Y.value, n3.Y.value))

        m = self.create_parabola_model()
        Y = m.Y.value
        n5 = DataScaler(m, self.domain, normalize_Y=False)
        n5.output_transform = GPflowOpt.transforms.LinearTransform(2, 0)
        self.assertTrue(np.allclose(m.X.value, scaledX))
        self.assertTrue(np.allclose(n5.Y.value, Y))
        self.assertTrue(np.allclose(m.Y.value, Y*2))

    def test_predict_scaling(self):
        m = self.create_parabola_model()
        n = DataScaler(self.create_parabola_model(), self.domain)
        m.optimize()
        n.optimize()

        Xt = GPflowOpt.design.RandomDesign(20, self.domain).generate()
        fr, vr = m.predict_f(Xt)
        fs, vs = n.predict_f(Xt)
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))
        self.assertTrue(np.allclose(vr, vs, atol=1e-3))

        fr, vr = m.predict_y(Xt)
        fs, vs = n.predict_y(Xt)
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))
        self.assertTrue(np.allclose(vr, vs, atol=1e-3))

        fr, vr = m.predict_f_full_cov(Xt)
        fs, vs = n.predict_f_full_cov(Xt)
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))
        self.assertTrue(np.allclose(vr, vs, atol=1e-3))

        Yt = parabola2d(Xt) + np.random.rand(20, 1) * 0.25
        fr = m.predict_density(Xt, Yt)
        fs = m.predict_density(Xt, Yt)
        self.assertTrue(np.allclose(fr, fs, atol=1e-3))
