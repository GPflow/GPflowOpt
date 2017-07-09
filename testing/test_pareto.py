import unittest
import numpy as np
import GPflowOpt


class TestPareto(unittest.TestCase):
    def test_update(self):
        scores = np.array(
            [[0.9575, 0.4218], [0.9649, 0.9157], [0.1576, 0.7922], [0.9706, 0.9595], [0.9572, 0.6557], [0.4854, 0.0357],
             [0.8003, 0.8491], [0.1419, 0.9340]])

        p = GPflowOpt.pareto.Pareto(scores)
        np.testing.assert_almost_equal(p.bounds.lb.value, np.array([[0, 0], [0, 2], [2, 0], [1, 2], [3, 0]]),
                                       err_msg='LBIDX incorrect.')
        np.testing.assert_almost_equal(p.bounds.ub.value, np.array([[2, 2], [1, 4], [3, 2], [2, 1], [4, 3]]),
                                       err_msg='UBIDX incorrect.')
        np.testing.assert_almost_equal(p.front.value, np.array([[0.1419, 0.9340], [0.1576, 0.7922], [0.4854, 0.0357]]),
                                       decimal=4, err_msg='PF incorrect.')
        np.testing.assert_almost_equal(p.idx_dom_augm.ub.value, np.array([[3, 3], [2, 5], [4, 3], [3, 2], [5, 4]]) - 1,
                                       err_msg='ubIdxDomAugm incorrect.')
        np.testing.assert_almost_equal(p.idx_dom_augm.lb.value, np.array([[1, 1], [1, 3], [3, 1], [2, 3], [4, 1]]) - 1,
                                       err_msg='lbIdxDomAugm incorrect.')

        np.testing.assert_almost_equal(p.hypervolume([2, 2]), 3.3878, decimal=2, err_msg='hypervolume incorrect.')
