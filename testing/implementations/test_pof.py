import pytest
import gpflowopt
import numpy as np


@pytest.fixture()
def pof(plane_model):
    yield gpflowopt.acquisition.ProbabilityOfFeasibility(plane_model)


def test_constraint_indices(pof):
    print(pof.constraint_indices())
    assert pof.constraint_indices() == np.arange(1, dtype=int)


def test_pof_validity(pof):
    X1 = np.random.rand(10, 2) / 4
    X2 = np.random.rand(10, 2) / 4 + 0.75
    assert np.all(pof.evaluate(X1) > 0.85)
    assert np.all(pof.evaluate(X2) < 0.15)
    assert np.all(pof.evaluate(X1) > pof.evaluate(X2).T)
