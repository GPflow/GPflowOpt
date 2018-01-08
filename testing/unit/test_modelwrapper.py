import gpflowopt
from gpflow import DataHolder, Parameterized, settings, params_as_tensors, autoflow, train
from gpflow.test_util import GPflowTestCase
import numpy as np
import contextlib
from ..utility import create_parabola_model


class MethodOverride(gpflowopt.params.ModelWrapper):

    def __init__(self, m):
        super(MethodOverride, self).__init__(m)
        self.A = DataHolder(np.array([1.0]))

    @autoflow((settings.tf_float, [None, None]))
    @params_as_tensors
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        m, v = self._build_predict(Xnew)
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


class TestModelWrapper(GPflowTestCase):

    def setUp(self):
        self.m = create_parabola_model(gpflowopt.domain.UnitCube(2))

    @contextlib.contextmanager
    def test_context(self):
        yield super(TestModelWrapper, self).test_context(graph=self.m.graph)

    def test_object_integrity(self):
        w = gpflowopt.params.ModelWrapper(self.m)
        assert w.wrapped == self.m
        assert self.m._parent == w
        assert w.compile == self.m.compile

    def test_optimize(self):
        with self.test_context():
            w = gpflowopt.params.ModelWrapper(self.m)
            logL = self.m.compute_log_likelihood()
            self.assertTrue(np.allclose(logL, w.compute_log_likelihood()))

            # Check if compiled & optimized, verify attributes are set in the right object.
            opt = train.ScipyOptimizer(options={'maxiter': 5})
            opt.minimize(w.wrapped)
            assert hasattr(self.m, '_likelihood_tensor')
            assert '_likelihood_tensor' not in w.__dict__
            assert self.m.compute_log_likelihood() > logL

    def test_af_storage_detection(self):
        with self.test_context():
            # Regression test for a bug with predict_f/predict_y... etc.
            x = np.random.rand(10,2)
            self.m.predict_f(x)
            assert hasattr(self.m, '_autoflow_predict_f')
            w = MethodOverride(self.m)
            assert not hasattr(w, '_autoflow_predict_f')
            w.predict_f(x)
            assert hasattr(w, '_autoflow_predict_f')

    def test_set_wrapped_attributes(self):
        # Regression test for setting certain keys in the right object
        w = gpflowopt.params.ModelWrapper(self.m)
        w.num_latent = 5
        assert 'num_latent' not in w.__dict__
        assert 'num_latent' in self.m.__dict__
        assert w.num_latent == 5
        assert self.m.num_latent == 5

    def test_double_wrap(self):
        with self.test_context():
            n = gpflowopt.params.ModelWrapper(MethodOverride(self.m))
            Xt = np.random.rand(10, 2)
            n.predict_f(Xt)
            assert '_autoflow_predict_f' not in n.__dict__
            assert '_autoflow_predict_f' in n.wrapped.__dict__
            assert '_autoflow_predict_f' not in n.wrapped.wrapped.__dict__

            n = MethodOverride(gpflowopt.params.ModelWrapper(self.m))
            n.clear()
            Xn = np.random.rand(10, 2)
            Yn = np.random.rand(10, 1)
            n.X = Xn
            n.Y = Yn
            np.testing.assert_allclose(Xn, n.wrapped.wrapped.X.read_value())
            np.testing.assert_allclose(Yn, n.wrapped.wrapped.Y.read_value())

            assert 'Y' not in n.wrapped.__dict__
            assert 'X' not in n.wrapped.__dict__

            n.foo = 5
            assert 'foo' in n.wrapped.__dict__
            assert 'foo' not in n.wrapped.wrapped.__dict__

    def test_name(self):
        with self.test_context():
            n = gpflowopt.params.ModelWrapper(self.m)
            assert n.name == 'ModelWrapper.modelwrapper'
            p = Parameterized()
            p.model = n
            assert n.name == 'model.modelwrapper'
            n = MethodOverride(create_parabola_model(gpflowopt.domain.UnitCube(2)))
            assert n.name == 'MethodOverride.methodoverride'
