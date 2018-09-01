import gpflowopt
from gpflow import DataHolder, Parameterized, settings, params_as_tensors, autoflow, train
import numpy as np
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


def test_object_integrity(parabola_model):
    w = gpflowopt.params.ModelWrapper(parabola_model)
    assert w.wrapped == parabola_model
    assert parabola_model._parent == w
    assert w.compile == parabola_model.compile


def test_optimize(parabola_model):
    w = gpflowopt.params.ModelWrapper(parabola_model)
    logL = parabola_model.compute_log_likelihood()
    np.testing.assert_allclose(logL, w.compute_log_likelihood())

    # Check if compiled & optimized, verify attributes are set in the right object.
    opt = train.ScipyOptimizer(options={'maxiter': 5})
    opt.minimize(w.wrapped)
    assert hasattr(parabola_model, '_likelihood_tensor')
    assert '_likelihood_tensor' not in w.__dict__
    assert parabola_model.compute_log_likelihood() > logL


def test_af_storage_detection(parabola_model):
    # Regression test for a bug with predict_f/predict_y... etc.
    x = np.random.rand(10,2)
    parabola_model.predict_f(x)
    assert hasattr(parabola_model, '_autoflow_predict_f')
    w = MethodOverride(parabola_model)
    assert not hasattr(w, '_autoflow_predict_f')
    w.predict_f(x)
    assert hasattr(w, '_autoflow_predict_f')


def test_set_wrapped_attributes(parabola_model):
    # Regression test for setting certain keys in the right object
    w = gpflowopt.params.ModelWrapper(parabola_model)
    w.num_latent = 5
    assert 'num_latent' not in w.__dict__
    assert 'num_latent' in parabola_model.__dict__
    assert w.num_latent == 5
    assert parabola_model.num_latent == 5


def test_double_wrap(parabola_model):
    n = gpflowopt.params.ModelWrapper(MethodOverride(parabola_model))
    Xt = np.random.rand(10, 2)
    n.predict_f(Xt)
    assert '_autoflow_predict_f' not in n.__dict__
    assert '_autoflow_predict_f' in n.wrapped.__dict__
    assert '_autoflow_predict_f' not in n.wrapped.wrapped.__dict__

    n = MethodOverride(gpflowopt.params.ModelWrapper(parabola_model))
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


def test_name(parabola_model):
    n = gpflowopt.params.ModelWrapper(parabola_model)
    assert n.name == 'ModelWrapper.modelwrapper'
    p = Parameterized()
    p.model = n
    assert n.name == 'ModelWrapper.modelwrapper'
    n = MethodOverride(create_parabola_model(gpflowopt.domain.UnitCube(2)))
    assert n.name == 'MethodOverride.methodoverride'
