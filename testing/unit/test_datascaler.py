import gpflow
import gpflowopt
import numpy as np
from gpflow.training import ScipyOptimizer
from gpflowopt.scaling import DataScaler
import pytest
from ..utility import parabola2d


def test_object_integrity(parabola_model):
    Xs, Ys = parabola_model.X.read_value(), parabola_model.Y.read_value()
    n = DataScaler(parabola_model)
    np.testing.assert_allclose(Xs, n.X.read_value())
    np.testing.assert_allclose(Ys, n.Y.read_value())


### SCALING TESTS ###
@pytest.fixture()
def normY(parabola_model):
    return (parabola_model.Y.read_value() - np.mean(parabola_model.Y.read_value(), axis=0)) / np.std(parabola_model.Y.read_value(), axis=0)


@pytest.fixture()
def scaledX(parabola_model):
    return (parabola_model.X.read_value() + 1) / 2


def test_scaler_no_scaling(domain, parabola_model, scaledX, normY):
    n = DataScaler(parabola_model, normalize_Y=False)
    assert not n.normalize_output
    np.testing.assert_allclose(parabola_model.X.read_value(), n.X.read_value())
    np.testing.assert_allclose(parabola_model.Y.read_value(), n.Y.read_value())
    n.set_input_transform(domain >> gpflowopt.domain.UnitCube(domain.size))
    np.testing.assert_allclose(parabola_model.X.read_value(), scaledX)
    np.testing.assert_allclose(parabola_model.Y.read_value(), n.Y.read_value())
    n.normalize_output = True
    assert n.normalize_output
    np.testing.assert_allclose(parabola_model.Y.read_value(), normY)


def test_scaler_input_scaling(domain, parabola_model, scaledX, normY):
    origX = parabola_model.X.read_value()
    n = DataScaler(parabola_model, domain, normalize_Y=False)
    np.testing.assert_allclose(parabola_model.X.read_value(), scaledX)
    np.testing.assert_allclose(parabola_model.Y.read_value(), n.Y.read_value())
    n.normalize_output = True
    np.testing.assert_allclose(parabola_model.Y.read_value(), normY)
    n.set_input_transform(gpflowopt.domain.UnitCube(domain.size) >> gpflowopt.domain.UnitCube(domain.size))
    np.testing.assert_allclose(parabola_model.X.read_value(), origX)


def test_scaler_output_scaling(parabola_model, normY):
    n = DataScaler(parabola_model, normalize_Y=True)
    np.testing.assert_allclose(parabola_model.X.read_value(), n.X.read_value())
    np.testing.assert_allclose(parabola_model.Y.read_value(), normY)
    n.normalize_output = False
    np.testing.assert_allclose(parabola_model.Y.read_value(), n.Y.read_value())


def test_scaler_all_scaling(domain, parabola_model, scaledX, normY):
    origY = parabola_model.Y.read_value()
    n = DataScaler(parabola_model, domain, normalize_Y=True)
    np.testing.assert_allclose(parabola_model.X.read_value(), scaledX)
    np.testing.assert_allclose(parabola_model.Y.read_value(), normY)
    n.normalize_output = False
    np.testing.assert_allclose(parabola_model.Y.read_value(), origY)


def test_scaler_misc(parabola_model, domain, scaledX):
    Y = parabola_model.Y.read_value()
    n = DataScaler(parabola_model, domain, normalize_Y=False)
    n.set_output_transform(gpflowopt.transforms.LinearTransform(2., 0.))
    np.testing.assert_allclose(parabola_model.X.read_value(), scaledX)
    np.testing.assert_allclose(n.Y.read_value(), Y)
    np.testing.assert_allclose(parabola_model.Y.read_value(), 2 * Y)


### PREDICTION TESTS ###
@pytest.fixture()
def predict_setup(domain, parabola_model):
    m = gpflow.models.GPR(parabola_model.X.read_value(), parabola_model.Y.read_value(), gpflow.kernels.RBF(2, ARD=True))
    n = DataScaler(m, domain, normalize_Y=True)
    opt = ScipyOptimizer()
    opt.minimize(parabola_model)
    opt.minimize(n.wrapped)
    yield (parabola_model, n)


@pytest.fixture(scope='module')
def Xt(domain):
    return gpflowopt.design.RandomDesign(20, domain).generate()


def test_predict_f(predict_setup, Xt):
    m, n = predict_setup
    fr, vr = m.predict_f(Xt)
    fs, vs = n.predict_f(Xt)
    np.testing.assert_allclose(fr, fs, atol=1e-3)
    np.testing.assert_allclose(vr, vs, atol=1e-3)


def test_predict_y(predict_setup, Xt):
    m, n = predict_setup
    fr, vr = m.predict_y(Xt)
    fs, vs = n.predict_y(Xt)
    np.testing.assert_allclose(fr, fs, atol=1e-3)
    np.testing.assert_allclose(vr, vs, atol=1e-3)


def test_predict_f_full_cov(predict_setup, Xt):
    m, n = predict_setup
    fr, vr = m.predict_f_full_cov(Xt)
    fs, vs = n.predict_f_full_cov(Xt)
    np.testing.assert_allclose(fr, fs, atol=1e-3)
    np.testing.assert_allclose(vr, vs, atol=1e-3)


def test_predict_density(predict_setup, Xt):
    m, n = predict_setup
    Yt = parabola2d(Xt)
    fr = m.predict_density(Xt, Yt)
    fs = n.predict_density(Xt, Yt)
    np.testing.assert_allclose(fr, fs, rtol=1e-2)
