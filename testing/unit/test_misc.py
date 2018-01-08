import copy
import gpflow
import gpflowopt
import numpy as np
from .test_acquisition import SimpleAcquisition


def test_randomize_model_no_prior(parabola_model):
    init_state = copy.deepcopy(parabola_model.read_trainables())
    gpflowopt.misc.randomize_model(parabola_model)
    for param_name, value in parabola_model.read_trainables().items():
        assert np.all(value != init_state[param_name])


def test_randomize_model_prior(parabola_model):
    parabola_model.clear()
    for p in parabola_model.parameters:
        p.prior = gpflow.priors.Uniform(0.2, 0.3)

    parabola_model.compile()
    gpflowopt.misc.randomize_model(parabola_model)
    for value in parabola_model.read_trainables().values():
        assert np.all(0.2 <= value)
        assert np.all(value <= 0.3)


def test_randomize_model_not_trainable(parabola_model):
    parabola_model.kern.variance.trainable = False
    gpflowopt.misc.randomize_model(parabola_model)
    assert np.allclose(parabola_model.kern.variance.read_value(), 1.)


def test_randomize_model_non_negative(parabola_model):
    gpflowopt.misc.randomize_model(parabola_model)
    for value in parabola_model.read_trainables().values():
        assert np.all(0 < value)


def test_hmc_eval_no_grad(parabola_model):
    acq = SimpleAcquisition(parabola_model)
    Xcand = np.random.rand(5, 2)
    result = gpflowopt.misc.hmc_eval(acq, Xcand, num_samples=5, gradients=False, epsilon=1e-6)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,1)
    np.testing.assert_array_almost_equal(acq.evaluate(Xcand), result, decimal=1)


def test_hmc_eval_grad(parabola_model):
    acq = SimpleAcquisition(parabola_model)
    Xcand = np.random.rand(5, 2)
    result = gpflowopt.misc.hmc_eval(acq, Xcand, num_samples=5, epsilon=1e-6)
    assert isinstance(result, tuple)
    assert result[0].shape == (5, 1)
    assert result[1].shape == (5, 2)
    map_result = acq.evaluate_with_gradients(Xcand)
    np.testing.assert_array_almost_equal(map_result[0], result[0], decimal=1)
