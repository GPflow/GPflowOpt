import unittest
import numpy as np
import GPflowOpt


class TestUtilities(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_nonDominatedSort(self):
        scores = np.array([[0.9575, 0.4218], [0.9649, 0.9157], [0.1576, 0.7922], [0.9706, 0.9595], [0.9572, 0.6557],
                           [0.4854, 0.0357], [0.8003, 0.8491], [0.1419, 0.9340]])
        d1, d2, d3 = GPflowOpt.pareto.non_dominated_sort(scores)
        np.testing.assert_almost_equal(d1, [5, 2, 7, 0, 4, 6, 1, 3], err_msg='Returned incorrect index.')
        np.testing.assert_almost_equal(d2, [1, 5, 0, 7, 1, 0, 2, 0], err_msg='Returned incorrect dominance')
        np.testing.assert_almost_equal(d3,
                                       np.array([0.2340, np.inf, 0.1427, np.inf, 0.2340, 0.8244, np.inf, 0.1427]),
                                       decimal=2, err_msg="Returned incorrect distance")

    def test_setdiffrows(self):
        points = np.random.rand(5, 3)
        result = GPflowOpt.pareto.setdiffrows(points, points)
        self.assertEqual(result.size, 0)
        self.assertEqual(result.shape, (0, 3))

        result = GPflowOpt.pareto.setdiffrows(np.vstack(([1, 2, 3], points)), points)
        self.assertEqual(result.size, 3)
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_almost_equal(result, np.array([[1, 2, 3]]))

    def test_uniquerows(self):
        points = np.random.rand(3, 3)
        result = GPflowOpt.pareto.unique_rows(np.vstack((points, points)))
        np.matrix.sort(points, axis=0)
        np.matrix.sort(points, axis=1)
        np.matrix.sort(result, axis=0)
        np.matrix.sort(result, axis=1)
        np.testing.assert_almost_equal(points, result)


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
        self.p = GPflowOpt.pareto.Pareto(objective_scores)

    def test_update(self):
        np.testing.assert_almost_equal(self.p.bounds.lb.value, np.array([[0, 0], [0, 2], [2, 0], [1, 2], [3, 0]]),
                                       err_msg='LBIDX incorrect.')
        np.testing.assert_almost_equal(self.p.bounds.ub.value, np.array([[2, 2], [1, 4], [3, 2], [2, 1], [4, 3]]),
                                       err_msg='UBIDX incorrect.')
        np.testing.assert_almost_equal(self.p.front.value, np.array([[0.1419, 0.9340], [0.1576, 0.7922],
                                                                     [0.4854, 0.0357]]), decimal=4,
                                       err_msg='PF incorrect.')
        np.testing.assert_almost_equal(self.p.idx_dom_augm.ub.value, np.array([[2, 2], [1, 4], [3, 2], [2, 1], [4, 3]]),
                                       err_msg='ubIdxDomAugm incorrect.')
        np.testing.assert_almost_equal(self.p.idx_dom_augm.lb.value, np.array([[0, 0], [0, 2], [2, 0], [1, 2], [3, 0]]),
                                       err_msg='lbIdxDomAugm incorrect.')

    def test_hypervolume(self):
        np.testing.assert_almost_equal(self.p.hypervolume([2, 2]), 3.3878, decimal=2, err_msg='hypervolume incorrect.')
