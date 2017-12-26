import copy
import gpflow
import gpflowopt
import numpy as np


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
    for param_name, value in parabola_model.read_trainables().items():
        assert np.all(0.2 <= value)
        assert np.all(value <= 0.3)


def test_randomize_model_not_trainable(parabola_model):
    parabola_model.kern.variance.trainable = False
    gpflowopt.misc.randomize_model(parabola_model)
    assert np.allclose(parabola_model.kern.variance.read_value(), 1.)
