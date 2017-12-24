import pytest
import gpflowopt
import numpy as np
from ..utility import parabola2d


@pytest.fixture()
def ei(parabola_model):
    yield gpflowopt.acquisition.ExpectedImprovement(parabola_model)


def test_objective_indices(ei):
    np.testing.assert_almost_equal(ei.objective_indices(), np.arange(1, dtype=int))


def test_setup(ei):
    ei._optimize_models()
    ei._setup()
    fmin = np.min(ei.data[1])
    assert ei.fmin.read_value() > 0
    np.testing.assert_allclose(ei.fmin.read_value(), fmin, atol=1e-2)

    p = np.array([[0.0, 0.0]])
    ei.set_data(np.vstack((ei.data[0], p)), np.vstack((ei.data[1], parabola2d(p))))
    ei._optimize_models()
    ei._setup()
    np.testing.assert_allclose(ei.fmin.read_value(), 0, atol=1e-2)


def test_validity(ei):
    Xcenter = np.random.rand(20, 2) * 0.25 - 0.125
    X = np.random.rand(100, 2) * 2 - 1
    hor_idx = np.abs(X[:, 0]) > 0.8
    ver_idx = np.abs(X[:, 1]) > 0.8
    Xborder = np.vstack((X[hor_idx, :], X[ver_idx, :]))
    ei1 = ei.evaluate(Xborder)
    ei2 = ei.evaluate(Xcenter)
    assert np.min(ei2) > np.max(ei1)
    assert np.all(ei.feasible_data_index())
