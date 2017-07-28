import unittest
import numpy as np
import GPflowOpt


class TestUtilities(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_nonDominatedSort(self):
        scores = np.array([[0.9575, 0.4218], [0.9649, 0.9157], [0.1576, 0.7922], [0.9706, 0.9595], [0.9572, 0.6557],
                           [0.4854, 0.0357], [0.8003, 0.8491], [0.1419, 0.9340]])
        d1, d2 = GPflowOpt.pareto.non_dominated_sort(scores)
        np.testing.assert_almost_equal(d1, [[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934 ]], err_msg='Returned incorrect Pareto set.')
        np.testing.assert_almost_equal(d2, [1, 5, 0, 7, 1, 0, 2, 0], err_msg='Returned incorrect dominance')


class TestPareto(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        objective_scores = np.array([[0.9575, 0.4218],
                                     [0.9649, 0.9157],
                                     [0.1576, 0.7922],
                                     [0.9706, 0.9595],
                                     [0.9572, 0.6557],
                                     [0.4854, 0.0357],
                                     [0.8003, 0.8491],
                                     [0.1419, 0.9340]])
        self.p_2d = GPflowOpt.pareto.Pareto(objective_scores)
        self.p_generic = GPflowOpt.pareto.Pareto(np.zeros((1, 2)))
        self.p_generic.update(objective_scores, generic_strategy=True)

    def test_update(self):
        np.testing.assert_almost_equal(self.p_2d.bounds.lb.value, np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
                                       err_msg='LBIDX incorrect.')
        np.testing.assert_almost_equal(self.p_2d.bounds.ub.value, np.array([[1, 4], [2, 1], [3, 2], [4, 3]]),
                                       err_msg='UBIDX incorrect.')
        np.testing.assert_almost_equal(self.p_2d.front.value, np.array([[0.1419, 0.9340], [0.1576, 0.7922],
                                                                     [0.4854, 0.0357]]), decimal=4,
                                       err_msg='PF incorrect.')

        np.testing.assert_almost_equal(self.p_generic.bounds.lb.value, np.array([[3, 0], [2, 0], [1, 2], [0, 2], [0, 0]]),
                                       err_msg='LBIDX incorrect.')
        np.testing.assert_almost_equal(self.p_generic.bounds.ub.value, np.array([[4, 3], [3, 2], [2, 1], [1, 4], [2, 2]]),
                                       err_msg='UBIDX incorrect.')
        np.testing.assert_almost_equal(self.p_generic.front.value, np.array([[0.1419, 0.9340], [0.1576, 0.7922],
                                                                 [0.4854, 0.0357]]), decimal=4,
                                       err_msg='PF incorrect.')

        self.assertFalse(np.array_equal(self.p_2d.bounds.lb, self.p_generic.bounds.lb), msg='Cell lowerbounds are exactly the same for all strategies.')
        self.assertFalse(np.array_equal(self.p_2d.bounds.ub, self.p_generic.bounds.ub), msg='Cell upperbounds are exactly the same for all strategies.')

    def test_hypervolume(self):
        np.testing.assert_almost_equal(self.p_2d.hypervolume([2, 2]), 3.3878, decimal=2, err_msg='hypervolume incorrect.')
        np.testing.assert_almost_equal(self.p_generic.hypervolume([2, 2]), 3.3878, decimal=2, err_msg='hypervolume incorrect.')

        np.testing.assert_almost_equal(self.p_2d.hypervolume([1, 1]), self.p_generic.hypervolume([1, 1]), decimal=20,
                                       err_msg='hypervolume of different strategies incorrect.')
