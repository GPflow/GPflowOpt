import pytest
import gpflowopt
import numpy as np
from ..utility import parabola2d


@pytest.fixture()
def poi(parabola_model):
    yield gpflowopt.acquisition.ProbabilityOfImprovement(parabola_model)


def test_objective_indices(poi):
    np.testing.assert_almost_equal(poi.objective_indices(), np.arange(1, dtype=int))


def test_setup(poi):
    poi._optimize_models()
    poi._setup()
    fmin = np.min(poi.data[1])
    assert poi.fmin.read_value() > 0
    np.testing.assert_allclose(poi.fmin.read_value(), fmin, atol=1e-2)

    p = np.array([[0.0, 0.0]])
    poi.set_data(np.vstack((poi.data[0], p)), np.vstack((poi.data[1], parabola2d(p))))
    poi._optimize_models()
    poi._setup()
    np.testing.assert_allclose(poi.fmin.read_value(), 0, atol=1e-2)


def test_validity(poi):
    Xcenter = np.random.rand(20, 2) * 0.25 - 0.125
    X = np.random.rand(100, 2) * 2 - 1
    hor_idx = np.abs(X[:, 0]) > 0.8
    ver_idx = np.abs(X[:, 1]) > 0.8
    Xborder = np.vstack((X[hor_idx, :], X[ver_idx, :]))
    poi1 = poi.evaluate(Xborder)
    poi2 = poi.evaluate(Xcenter)
    assert np.min(poi2) > np.max(poi1)
    assert np.all(poi.feasible_data_index())
