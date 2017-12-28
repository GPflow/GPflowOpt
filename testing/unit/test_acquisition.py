import gpflowopt
import numpy as np
import gpflow
import tensorflow as tf
import pytest
from gpflow.test_util import GPflowTestCase
from ..utility import create_parabola_model, parabola2d, plane
import contextlib
import copy


class SimpleAcquisition(gpflowopt.acquisition.Acquisition):
    def __init__(self, model):
        super(SimpleAcquisition, self).__init__(model)
        self.counter = 0

    def _setup(self):
        super(SimpleAcquisition, self)._setup()
        self.counter += 1

    def _build_acquisition(self, Xcand):
        return self.models[0]._build_predict(Xcand)[0]

    def read_trainables(self, session=None):
        # TODO: @javdrher to be removed when gpflow performs this copy
        return copy.deepcopy(super(SimpleAcquisition, self).read_trainables(session))


def compare_model_state(sa, sb):
    eq_keys = list(sa.keys()) == list(sb.keys())
    a = np.hstack(v for v in sa.values())
    b = np.hstack(v for v in sb.values())
    return eq_keys and np.allclose(a, b)


@pytest.fixture()
def acquisition(parabola_model):
    yield SimpleAcquisition(parabola_model)


def test_object_integrity(acquisition):
    assert len(acquisition.models) == 1


def test_setup_trigger(domain, acquisition):
    init_state = dict(
        zip(acquisition.models[0].read_trainables().keys(), (np.array(1.0), np.array([ 1.,  1.]), np.array(1.0)))
    )
    assert compare_model_state(init_state, acquisition.read_trainables())
    assert acquisition._needs_setup
    assert acquisition.counter == 0
    acquisition.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())
    assert not acquisition._needs_setup
    assert acquisition.counter == 1
    assert not compare_model_state(init_state, acquisition.read_trainables())

    acquisition._needs_setup = True
    acquisition.models[0].assign(init_state)
    acquisition.evaluate_with_gradients(gpflowopt.design.RandomDesign(10, domain).generate())
    assert not acquisition._needs_setup
    assert acquisition.counter == 2


def test_data(acquisition):
   with gpflow.params_as_tensors_for(acquisition):
       assert isinstance(acquisition.data[0], tf.Tensor)
       assert isinstance(acquisition.data[1], tf.Tensor)


def test_data_update(domain, acquisition):
    # Verify the effect of setting the data
    design = gpflowopt.design.RandomDesign(10, domain)
    X = np.vstack((acquisition.data[0], design.generate()))
    Y = parabola2d(X)
    acquisition._needs_setup = False
    acquisition.set_data(X, Y)
    np.testing.assert_allclose(acquisition.data[0], X, atol=1e-5, err_msg="Samples not updated")
    np.testing.assert_allclose(acquisition.data[1], Y, atol=1e-5, err_msg="Values not updated")
    assert acquisition._needs_setup


def test_data_indices(acquisition):
    # Return all data as feasible.
    assert acquisition.feasible_data_index().shape == (acquisition.data[0].shape[0],)


def test_enable_scaling(domain, acquisition):
    assert not any(m.wrapped.X.read_value() in gpflowopt.domain.UnitCube(domain.size) for m in acquisition.models)
    acquisition._needs_setup = False
    acquisition.enable_scaling(domain)
    assert all(m.wrapped.X.read_value() in gpflowopt.domain.UnitCube(domain.size) for m in acquisition.models)
    assert acquisition._needs_setup


def test_result_shape_tf(acquisition):
    # Verify the returned shape of evaluate
    x_tf = tf.placeholder(tf.float64, shape=(50, 2))
    tens = acquisition._build_acquisition(x_tf)
    assert isinstance(tens, tf.Tensor)


def test_result_shape_np(domain, acquisition):
    design = gpflowopt.design.RandomDesign(50, domain)
    res = acquisition.evaluate(design.generate())
    assert res.shape == (50, 1)
    res = acquisition.evaluate_with_gradients(design.generate())
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res[0].shape == (50, 1)
    assert res[1].shape == (50, domain.size)


def test_optimize(acquisition):
    acquisition.optimize_restarts = 0
    state = acquisition.read_trainables()
    acquisition._optimize_models()
    assert compare_model_state(state, acquisition.read_trainables())

    acquisition.optimize_restarts = 1
    acquisition._optimize_models()
    assert not compare_model_state(state, acquisition.read_trainables())
