import pytest
import gpflowopt
import numpy as np


@pytest.fixture()
def mes(domain, parabola_model):
    yield gpflowopt.acquisition.MinValueEntropySearch(parabola_model, domain)


def test_objective_indices(mes):
    assert mes.objective_indices() == np.arange(1, dtype=int)


def test_setup(mes):
    fmin = np.min(mes.data[1])
    assert fmin > 0
    assert mes.samples.shape == (mes.num_samples,)


def test_mes_validity(mes):
    Xcenter = np.random.rand(20, 2) * 0.25 - 0.125
    X = np.random.rand(100, 2) * 2 - 1
    hor_idx = np.abs(X[:, 0]) > 0.8
    ver_idx = np.abs(X[:, 1]) > 0.8
    Xborder = np.vstack((X[hor_idx, :], X[ver_idx, :]))
    mes1 = mes.evaluate(Xborder)
    mes2 = mes.evaluate(Xcenter)
    assert np.min(mes2) + 1e-6 > np.max(mes1)
    assert np.all(mes.feasible_data_index())
