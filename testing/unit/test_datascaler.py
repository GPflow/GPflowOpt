import gpflowopt
import numpy as np
from gpflow.training import ScipyOptimizer
from gpflowopt.scaling import DataScaler
import pytest
import tensorflow as tf
from ..utility import create_parabola_model, parabola2d


@pytest.fixture(scope='module')
def domain():
    return np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])


@pytest.fixture()
def model(domain):
    with tf.Session(graph=tf.Graph()):
        yield create_parabola_model(domain)


def test_object_integrity(model):
    Xs, Ys = model.X.read_value(), model.Y.read_value()
    n = DataScaler(model)
    np.testing.assert_allclose(Xs, n.X.read_value())
    np.testing.assert_allclose(Ys, n.Y.read_value())


### SCALING TESTS ###
@pytest.fixture()
def normY(model):
    return (model.Y.read_value() - np.mean(model.Y.read_value(), axis=0)) / np.std(model.Y.read_value(), axis=0)


@pytest.fixture()
def scaledX(model):
    return (model.X.read_value() + 1) / 2


def test_scaler_no_scaling(domain, model, scaledX, normY):
    n = DataScaler(model, normalize_Y=False)
    assert not n.normalize_output
    np.testing.assert_allclose(model.X.read_value(), n.X.read_value())
    np.testing.assert_allclose(model.Y.read_value(), n.Y.read_value())
    n.set_input_transform(domain >> gpflowopt.domain.UnitCube(domain.size))
    np.testing.assert_allclose(model.X.read_value(), scaledX)
    np.testing.assert_allclose(model.Y.read_value(), n.Y.read_value())
    n.normalize_output = True
    assert n.normalize_output
    np.testing.assert_allclose(model.Y.read_value(), normY)


def test_scaler_input_scaling(domain, model, scaledX, normY):
    origX = model.X.read_value()
    n = DataScaler(model, domain, normalize_Y=False)
    np.testing.assert_allclose(model.X.read_value(), scaledX)
    np.testing.assert_allclose(model.Y.read_value(), n.Y.read_value())
    n.normalize_output = True
    np.testing.assert_allclose(model.Y.read_value(), normY)
    n.set_input_transform(gpflowopt.domain.UnitCube(domain.size) >> gpflowopt.domain.UnitCube(domain.size))
    np.testing.assert_allclose(model.X.read_value(), origX)


def test_scaler_output_scaling(model, normY):
    n = DataScaler(model, normalize_Y=True)
    np.testing.assert_allclose(model.X.read_value(), n.X.read_value())
    np.testing.assert_allclose(model.Y.read_value(), normY)
    n.normalize_output = False
    np.testing.assert_allclose(model.Y.read_value(), n.Y.read_value())


def test_scaler_all_scaling(domain, model, scaledX, normY):
    origY = model.Y.read_value()
    n = DataScaler(model, domain, normalize_Y=True)
    np.testing.assert_allclose(model.X.read_value(), scaledX)
    np.testing.assert_allclose(model.Y.read_value(), normY)
    n.normalize_output = False
    np.testing.assert_allclose(model.Y.read_value(), origY)


def test_scaler_misc(model, domain, scaledX):
    Y = model.Y.read_value()
    n = DataScaler(model, domain, normalize_Y=False)
    n.set_output_transform(gpflowopt.transforms.LinearTransform(2., 0.))
    np.testing.assert_allclose(model.X.read_value(), scaledX)
    np.testing.assert_allclose(n.Y.read_value(), Y)
    np.testing.assert_allclose(model.Y.read_value(), 2*Y)


### PREDICTION TESTS ###
@pytest.fixture(scope='module')
def predict_setup(domain):
    with tf.Session(graph=tf.Graph()):
        m = create_parabola_model(domain)
        n = DataScaler(create_parabola_model(domain), domain, normalize_Y=True)
        opt = ScipyOptimizer()
        opt.minimize(m)
        opt.minimize(n.wrapped)
        yield (m, n)


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
