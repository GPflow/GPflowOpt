import gpflowopt
import gpflow
import numpy as np
from ..utility import create_parabola_model, GPflowOptTestCase

float_type = gpflow.settings.dtypes.float_type


class MethodOverride(gpflowopt.models.ModelWrapper):

    def __init__(self, m):
        super(MethodOverride, self).__init__(m)
        self.A = gpflow.param.DataHolder(np.array([1.0]))

    @gpflow.param.AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        m, v = self.build_predict(Xnew)
        return self.A * m, v

    @property
    def X(self):
        return self.wrapped.X

    @X.setter
    def X(self, Xc):
        self.wrapped.X = Xc

    @property
    def foo(self):
        return 1

    @foo.setter
    def foo(self, val):
        self.wrapped.foo = val


class TestModelWrapper(GPflowOptTestCase):

    def setUp(self):
        self.m = create_parabola_model(gpflowopt.domain.UnitCube(2))

    def test_object_integrity(self):
        w = gpflowopt.models.ModelWrapper(self.m)
        self.assertEqual(w.wrapped, self.m)
        self.assertEqual(self.m._parent, w)
        self.assertEqual(w.optimize, self.m.optimize)

    def test_optimize(self):
        with self.test_session():
            w = gpflowopt.models.ModelWrapper(self.m)
            logL = self.m.compute_log_likelihood()
            self.assertTrue(np.allclose(logL, w.compute_log_likelihood()))

            # Check if compiled & optimized, verify attributes are set in the right object.
            w.optimize(maxiter=5)
            self.assertTrue(hasattr(self.m, '_minusF'))
            self.assertFalse('_minusF' in w.__dict__)
            self.assertGreater(self.m.compute_log_likelihood(), logL)

    def test_af_storage_detection(self):
        with self.test_session():
            # Regression test for a bug with predict_f/predict_y... etc.
            x = np.random.rand(10,2)
            self.m.predict_f(x)
            self.assertTrue(hasattr(self.m, '_predict_f_AF_storage'))
            w = MethodOverride(self.m)
            self.assertFalse(hasattr(w, '_predict_f_AF_storage'))
            w.predict_f(x)
            self.assertTrue(hasattr(w, '_predict_f_AF_storage'))

    def test_set_wrapped_attributes(self):
        # Regression test for setting certain keys in the right object
        w = gpflowopt.models.ModelWrapper(self.m)
        w._needs_recompile = False
        self.assertFalse('_needs_recompile' in w.__dict__)
        self.assertTrue('_needs_recompile' in self.m.__dict__)
        self.assertFalse(w._needs_recompile)
        self.assertFalse(self.m._needs_recompile)

    def test_double_wrap(self):
        with self.test_session():
            n = gpflowopt.models.ModelWrapper(MethodOverride(self.m))
            n.optimize(maxiter=10)
            Xt = np.random.rand(10, 2)
            n.predict_f(Xt)
            self.assertFalse('_predict_f_AF_storage' in n.__dict__)
            self.assertTrue('_predict_f_AF_storage' in n.wrapped.__dict__)
            self.assertFalse('_predict_f_AF_storage' in n.wrapped.wrapped.__dict__)

            n = MethodOverride(gpflowopt.models.ModelWrapper(self.m))
            Xn = np.random.rand(10, 2)
            Yn = np.random.rand(10, 1)
            n.X = Xn
            n.Y = Yn
            self.assertTrue(np.allclose(Xn, n.wrapped.wrapped.X.value))
            self.assertTrue(np.allclose(Yn, n.wrapped.wrapped.Y.value))
            self.assertFalse('Y' in n.wrapped.__dict__)
            self.assertFalse('X' in n.wrapped.__dict__)

            n.foo = 5
            self.assertTrue('foo' in n.wrapped.__dict__)
            self.assertFalse('foo' in n.wrapped.wrapped.__dict__)

    def test_name(self):
        with self.test_session():
            n = gpflowopt.models.ModelWrapper(self.m)
            self.assertEqual(n.name, 'unnamed.modelwrapper')
            p = gpflow.param.Parameterized()
            p.model = n
            self.assertEqual(n.name, 'model.modelwrapper')
            n = MethodOverride(create_parabola_model(gpflowopt.domain.UnitCube(2)))
            self.assertEqual(n.name, 'unnamed.methodoverride')

    def test_parent_hook(self):
        with self.test_session():
            self.m.optimize(maxiter=5)
            w = gpflowopt.models.ModelWrapper(self.m)
            self.assertTrue(isinstance(self.m.highest_parent, gpflowopt.models.ParentHook))
            self.assertEqual(self.m.highest_parent._hp, w)
            self.assertEqual(self.m.highest_parent._hm, w)

            w2 = gpflowopt.models.ModelWrapper(w)
            self.assertEqual(self.m.highest_parent._hp, w2)
            self.assertEqual(self.m.highest_parent._hm, w2)

            p = gpflow.param.Parameterized()
            p.model = w2
            self.assertEqual(self.m.highest_parent._hp, p)
            self.assertEqual(self.m.highest_parent._hm, w2)

            p.predictor = create_parabola_model(gpflowopt.domain.UnitCube(2))
            p.predictor.predict_f(p.predictor.X.value)
            self.assertTrue(hasattr(p.predictor, '_predict_f_AF_storage'))
            self.assertFalse(self.m._needs_recompile)
            self.m.highest_parent._needs_recompile = True
            self.assertFalse('_needs_recompile' in p.__dict__)
            self.assertFalse('_needs_recompile' in w.__dict__)
            self.assertFalse('_needs_recompile' in w2.__dict__)
            self.assertTrue(self.m._needs_recompile)
            self.assertFalse(hasattr(p.predictor, '_predict_f_AF_storage'))

            self.assertEqual(self.m.highest_parent.get_free_state, p.get_free_state)
            self.m.highest_parent._needs_setup = True
            self.assertTrue(hasattr(p, '_needs_setup'))
            self.assertTrue(p._needs_setup)

