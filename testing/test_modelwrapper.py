import GPflowOpt
import unittest
import GPflow
import numpy as np

float_type = GPflow.settings.dtypes.float_type


class MethodOverride(GPflowOpt.models.ModelWrapper):

    def __init__(self, m):
        super(MethodOverride, self).__init__(m)
        self.A = GPflow.param.DataHolder(np.array([1.0]))

    @GPflow.param.AutoFlow((float_type, [None, None]))
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


class TestModelWrapper(unittest.TestCase):

    def simple_model(self):
        x = np.random.rand(10,2) * 2 * np.pi
        y = np.sin(x[:,[0]])
        m = GPflow.gpr.GPR(x,y, kern=GPflow.kernels.RBF(1))
        return m

    def test_object_integrity(self):
        m = self.simple_model()
        w = GPflowOpt.models.ModelWrapper(m)
        self.assertEqual(w.wrapped, m)
        self.assertEqual(m._parent, w)
        self.assertEqual(w.optimize, m.optimize)

    def test_optimize(self):
        m = self.simple_model()
        w = GPflowOpt.models.ModelWrapper(m)
        logL = m.compute_log_likelihood()
        self.assertTrue(np.allclose(logL, w.compute_log_likelihood()))

        # Check if compiled & optimized, verify attributes are set in the right object.
        w.optimize(maxiter=5)
        self.assertTrue(hasattr(m, '_minusF'))
        self.assertFalse('_minusF' in w.__dict__)
        self.assertGreater(m.compute_log_likelihood(), logL)

    def test_af_storage_detection(self):
        # Regression test for a bug with predict_f/predict_y... etc.
        m = self.simple_model()
        x = np.random.rand(10,2)
        m.predict_f(x)
        self.assertTrue(hasattr(m, '_predict_f_AF_storage'))
        w = MethodOverride(m)
        self.assertFalse(hasattr(w, '_predict_f_AF_storage'))
        w.predict_f(x)
        self.assertTrue(hasattr(w, '_predict_f_AF_storage'))

    def test_set_wrapped_attributes(self):
        # Regression test for setting certain keys in the right object
        m = self.simple_model()
        w = GPflowOpt.models.ModelWrapper(m)
        w._needs_recompile = False
        self.assertFalse('_needs_recompile' in w.__dict__)
        self.assertTrue('_needs_recompile' in m.__dict__)
        self.assertFalse(w._needs_recompile)
        self.assertFalse(m._needs_recompile)

    def test_double_wrap(self):
        m = self.simple_model()
        n = GPflowOpt.models.ModelWrapper(MethodOverride(m))
        n.optimize(maxiter=10)
        Xt = np.random.rand(10, 2)
        n.predict_f(Xt)
        self.assertFalse('_predict_f_AF_storage' in n.__dict__)
        self.assertTrue('_predict_f_AF_storage' in n.wrapped.__dict__)
        self.assertFalse('_predict_f_AF_storage' in n.wrapped.wrapped.__dict__)

        n = MethodOverride(GPflowOpt.models.ModelWrapper(m))
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
        n = GPflowOpt.models.ModelWrapper(self.simple_model())
        self.assertEqual(n.name, 'unnamed.modelwrapper')
        p = GPflow.param.Parameterized()
        p.model = n
        self.assertEqual(n.name, 'model.modelwrapper')
        n = MethodOverride(self.simple_model())
        self.assertEqual(n.name, 'unnamed.methodoverride')

    def test_parent_hook(self):
        m = self.simple_model()
        m.optimize(maxiter=5)
        w = GPflowOpt.models.ModelWrapper(m)
        self.assertTrue(isinstance(m.highest_parent, GPflowOpt.models.ParentHook))
        self.assertEqual(m.highest_parent._hp, w)
        self.assertEqual(m.highest_parent._hm, w)

        w2 = GPflowOpt.models.ModelWrapper(w)
        self.assertEqual(m.highest_parent._hp, w2)
        self.assertEqual(m.highest_parent._hm, w2)

        p = GPflow.param.Parameterized()
        p.model = w2
        self.assertEqual(m.highest_parent._hp, p)
        self.assertEqual(m.highest_parent._hm, w2)

        p.predictor = self.simple_model()
        p.predictor.predict_f(p.predictor.X.value)
        self.assertTrue(hasattr(p.predictor, '_predict_f_AF_storage'))
        self.assertFalse(m._needs_recompile)
        m.highest_parent._needs_recompile = True
        self.assertFalse('_needs_recompile' in p.__dict__)
        self.assertFalse('_needs_recompile' in w.__dict__)
        self.assertFalse('_needs_recompile' in w2.__dict__)
        self.assertTrue(m._needs_recompile)
        self.assertFalse(hasattr(p.predictor, '_predict_f_AF_storage'))


