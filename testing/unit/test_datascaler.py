import gpflowopt
import numpy as np
from gpflowopt.scaling import DataScaler
from ..utility import GPflowOptTestCase, create_parabola_model, parabola2d


class TestDataScaler(GPflowOptTestCase):

    @property
    def domain(self):
        return np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])

    def test_object_integrity(self):
        with self.test_session():
            m = create_parabola_model(self.domain)
            Xs, Ys = m.X.value, m.Y.value
            n = DataScaler(m, self.domain)

            self.assertTrue(np.allclose(Xs, n.X.value))
            self.assertTrue(np.allclose(Ys, n.Y.value))

    def test_enabling_transforms(self):
        with self.test_session():
            m = create_parabola_model(self.domain)
            normY = (m.Y.value - np.mean(m.Y.value, axis=0)) / np.std(m.Y.value, axis=0)
            scaledX = (m.X.value + 1) / 2

            n1 = DataScaler(m, normalize_Y=False)
            self.assertFalse(n1.normalize_output)
            self.assertTrue(np.allclose(m.X.value, n1.X.value))
            self.assertTrue(np.allclose(m.Y.value, n1.Y.value))
            n1.input_transform = self.domain >> gpflowopt.domain.UnitCube(self.domain.size)
            self.assertTrue(np.allclose(m.X.value, scaledX))
            self.assertTrue(np.allclose(m.Y.value, n1.Y.value))
            n1.normalize_output = True
            self.assertTrue(n1.normalize_output)
            self.assertTrue(np.allclose(m.Y.value, normY))

            m = create_parabola_model(self.domain)
            n2 = DataScaler(m, self.domain, normalize_Y=False)
            self.assertTrue(np.allclose(m.X.value, scaledX))
            self.assertTrue(np.allclose(m.Y.value, n2.Y.value))
            n2.normalize_output = True
            self.assertTrue(np.allclose(m.Y.value, normY))
            n2.input_transform = gpflowopt.domain.UnitCube(self.domain.size) >> gpflowopt.domain.UnitCube(self.domain.size)
            self.assertTrue(np.allclose(m.X.value, n1.X.value))

            m = create_parabola_model(self.domain)
            n3 = DataScaler(m, normalize_Y=True)
            self.assertTrue(np.allclose(m.X.value, n3.X.value))
            self.assertTrue(np.allclose(m.Y.value, normY))
            n3.normalize_output = False
            self.assertTrue(np.allclose(m.Y.value, n3.Y.value))

            m = create_parabola_model(self.domain)
            n4 = DataScaler(m, self.domain, normalize_Y=True)
            self.assertTrue(np.allclose(m.X.value, scaledX))
            self.assertTrue(np.allclose(m.Y.value, normY))
            n4.normalize_output = False
            self.assertTrue(np.allclose(m.Y.value, n3.Y.value))

            m = create_parabola_model(self.domain)
            Y = m.Y.value
            n5 = DataScaler(m, self.domain, normalize_Y=False)
            n5.output_transform = gpflowopt.transforms.LinearTransform(2, 0)
            self.assertTrue(np.allclose(m.X.value, scaledX))
            self.assertTrue(np.allclose(n5.Y.value, Y))
            self.assertTrue(np.allclose(m.Y.value, Y*2))

    def test_predict_scaling(self):
        with self.test_session():
            m = create_parabola_model(self.domain)
            n = DataScaler(create_parabola_model(self.domain), self.domain, normalize_Y=True)
            m.optimize()
            n.optimize()

            Xt = gpflowopt.design.RandomDesign(20, self.domain).generate()
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

            Yt = parabola2d(Xt)
            fr = m.predict_density(Xt, Yt)
            fs = n.predict_density(Xt, Yt)
            np.testing.assert_allclose(fr, fs, rtol=1e-2)


