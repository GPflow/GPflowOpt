import unittest
import GPflowOpt
import numpy as np
from .utility import create_parabola_model, create_plane_model, create_vlmop2_model, parabola2d, load_data


class TestExpectedImprovement(unittest.TestCase):
    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_parabola_model(self.domain)
        self.acquisition = GPflowOpt.acquisition.ExpectedImprovement(self.model)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="ExpectedImprovement returns all objectives")

    def test_setup(self):
        self.acquisition._optimize_models()
        self.acquisition.setup()
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.acquisition.data[0], p)),
                                  np.vstack((self.acquisition.data[1], parabola2d(p))))
        self.acquisition._optimize_models()
        self.acquisition.setup()
        self.assertTrue(np.allclose(self.acquisition.fmin.value, 0, atol=1e-1), msg="fmin not updated")

    def test_EI_validity(self):
        Xcenter = np.random.rand(20, 2) * 0.25 - 0.125
        X = np.random.rand(100, 2) * 2 - 1
        hor_idx = np.abs(X[:, 0]) > 0.8
        ver_idx = np.abs(X[:, 1]) > 0.8
        Xborder = np.vstack((X[hor_idx, :], X[ver_idx, :]))
        ei1 = self.acquisition.evaluate(Xborder)
        ei2 = self.acquisition.evaluate(Xcenter)
        self.assertGreater(np.min(ei2), np.max(ei1))
        self.assertTrue(np.all(self.acquisition.feasible_data_index()), msg="EI does never invalidate points")


class TestProbabilityOfImprovement(unittest.TestCase):
    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_parabola_model(self.domain)
        self.acquisition = GPflowOpt.acquisition.ProbabilityOfImprovement(self.model)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="PoI returns all objectives")

    def test_setup(self):
        self.acquisition._optimize_models()
        self.acquisition.setup()
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.acquisition.data[0], p)),
                                  np.vstack((self.acquisition.data[1], parabola2d(p))))
        self.acquisition._optimize_models()
        self.acquisition.setup()
        self.assertTrue(np.allclose(self.acquisition.fmin.value, 0, atol=1e-1), msg="fmin not updated")


class TestProbabilityOfFeasibility(unittest.TestCase):
    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_plane_model(self.domain)
        self.acquisition = GPflowOpt.acquisition.ProbabilityOfFeasibility(self.model)

    def test_constraint_indices(self):
        self.assertEqual(self.acquisition.constraint_indices(), np.arange(1, dtype=int),
                         msg="PoF returns all constraints")

    def test_PoF_validity(self):
        X1 = np.random.rand(10, 2) / 4
        X2 = np.random.rand(10, 2) / 4 + 0.75
        self.assertTrue(np.all(self.acquisition.evaluate(X1) > 0.85), msg="Left half of plane is feasible")
        self.assertTrue(np.all(self.acquisition.evaluate(X2) < 0.15), msg="Right half of plane is feasible")
        self.assertTrue(np.all(self.acquisition.evaluate(X1) > self.acquisition.evaluate(X2).T))


class TestLowerConfidenceBound(unittest.TestCase):
    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_plane_model(self.domain)
        self.acquisition = GPflowOpt.acquisition.LowerConfidenceBound(self.model, 3.2)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="LCB returns all objectives")

    def test_object_integrity(self):
        self.assertEqual(self.acquisition.sigma, 3.2)

    def test_LCB_validity(self):
        design = GPflowOpt.design.RandomDesign(200, self.domain).generate()
        q = self.acquisition.evaluate(design)
        p = self.acquisition.models[0].predict_f(design)[0]
        np.testing.assert_array_less(q, p)

    def test_LCB_validity_2(self):
        design = GPflowOpt.design.RandomDesign(200, self.domain).generate()
        self.acquisition.sigma = 0
        q = self.acquisition.evaluate(design)
        p = self.acquisition.models[0].predict_f(design)[0]
        np.testing.assert_allclose(q, p)


class TestHVProbabilityOfImprovement(unittest.TestCase):

    def setUp(self):
        self.model = create_vlmop2_model()
        self.data = load_data('vlmop.npz')
        self.acquisition = GPflowOpt.acquisition.HVProbabilityOfImprovement(self.model)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 2, msg="Model list has incorrect length.")
        for m1, m2 in zip(self.acquisition.models, self.model):
            self.assertEqual(m1, m2, msg="Incorrect model stored in ExpectedImprovement")
        self.assertEqual(len(self.acquisition._default_params), 2)
        for i in np.arange(2):
            self.assertTrue(np.allclose(np.sort(self.acquisition._default_params[i]), np.sort(np.array([0.5413] * 3)),
                                        atol=1e-2), msg="Initial hypers improperly stored")

    def test_HvPoI_validity(self):
        scores = self.acquisition.evaluate(self.data['candidates'])
        np.testing.assert_almost_equal(scores, self.data['scores'], decimal=2)
