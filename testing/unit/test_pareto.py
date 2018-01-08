import numpy as np
import gpflowopt
import pytest


@pytest.fixture(autouse=True)
def context(request, session):
    yield


def _objective_scores():
    return np.array([[0.9575, 0.4218], [0.9649, 0.9157], [0.1576, 0.7922], [0.9706, 0.9595], [0.9572, 0.6557],
                     [0.4854, 0.0357], [0.8003, 0.8491], [0.1419, 0.9340]])


pareto_configs = [(gpflowopt.pareto.Pareto, False,
                   np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
                   np.array([[1, 4], [2, 1], [3, 2], [4, 3]])),
                  (gpflowopt.pareto.Pareto, True,
                   np.array([[3, 0], [2, 0], [1, 2], [0, 2], [0, 0]]),
                   np.array([[4, 3], [3, 2], [2, 1], [1, 4], [2, 2]]))]


def test_non_dominated_sort():
    d1, d2 = gpflowopt.pareto.non_dominated_sort(_objective_scores())
    np.testing.assert_almost_equal(d1, [[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934 ]], err_msg='Returned incorrect Pareto set.')
    np.testing.assert_almost_equal(d2, [1, 5, 0, 7, 1, 0, 2, 0], err_msg='Returned incorrect dominance')


@pytest.mark.parametrize('pareto,generic,lb,ub', pareto_configs)
def test_update(pareto, generic, lb, ub):
    p_2d = pareto(np.zeros((1, 2)))
    p_2d.update(_objective_scores(), generic_strategy=generic)
    np.testing.assert_almost_equal(p_2d.bounds.lb.read_value(), lb)
    np.testing.assert_almost_equal(p_2d.bounds.ub.read_value(), ub)
    np.testing.assert_almost_equal(p_2d.front.read_value(), np.array([[0.1419, 0.9340], [0.1576, 0.7922],
                                                                      [0.4854, 0.0357]]), decimal=4)


@pytest.mark.parametrize('pareto,generic', [p[0:2] for p in pareto_configs])
def test_hypervolume(pareto, generic):
    p_2d = pareto(np.zeros((0, 2)))
    p_2d.update(_objective_scores(), generic_strategy=generic)
    np.testing.assert_almost_equal(p_2d.hypervolume([2, 2]), 3.3878, decimal=2, err_msg='hypervolume incorrect.')
