import unittest
import gpflowopt
import numpy as np
from parameterized import parameterized
from .utility import create_parabola_model, create_plane_model, create_vlmop2_model, parabola2d, load_data

domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
acquisitions = [gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)),
                gpflowopt.acquisition.ProbabilityOfImprovement(create_parabola_model(domain)),
                gpflowopt.acquisition.ProbabilityOfFeasibility(create_parabola_model(domain)),
                gpflowopt.acquisition.LowerConfidenceBound(create_parabola_model(domain)),
                gpflowopt.acquisition.HVProbabilityOfImprovement([create_parabola_model(domain),
                                                                  create_parabola_model(domain)]),
                gpflowopt.acquisition.MinValueEntropySearch(create_parabola_model(domain), domain)
                ]


class TestAcquisitionEvaluate(unittest.TestCase):

    _multiprocess_can_split_ = True

    @parameterized.expand(list(zip(acquisitions)))
    def test_evaluate(self, acquisition):
        X = gpflowopt.design.RandomDesign(10, domain).generate()
        p = acquisition.evaluate(X)
        self.assertTrue(isinstance(p, np.ndarray))
        self.assertTupleEqual(p.shape, (10, 1))

        q = acquisition.evaluate_with_gradients(X)
        self.assertTrue(isinstance(q, tuple))
        self.assertTrue(len(q), 2)
        for i in q:
            self.assertTrue(isinstance(i, np.ndarray))

        self.assertTupleEqual(q[0].shape, (10, 1))
        self.assertTrue(np.allclose(p, q[0]))
        self.assertTupleEqual(q[1].shape, (10, 2))


class TestExpectedImprovement(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_parabola_model(self.domain)
        self.acquisition = gpflowopt.acquisition.ExpectedImprovement(self.model)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="ExpectedImprovement returns all objectives")

    def test_setup(self):
        self.acquisition._optimize_models()
        self.acquisition._setup()
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.acquisition.data[0], p)),
                                  np.vstack((self.acquisition.data[1], parabola2d(p))))
        self.acquisition._optimize_models()
        self.acquisition._setup()
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

    _multiprocess_can_split_ = True

    def setUp(self):
        self.domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_parabola_model(self.domain)
        self.acquisition = gpflowopt.acquisition.ProbabilityOfImprovement(self.model)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="PoI returns all objectives")

    def test_setup(self):
        self.acquisition._optimize_models()
        self.acquisition._setup()
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.acquisition.data[0], p)),
                                  np.vstack((self.acquisition.data[1], parabola2d(p))))
        self.acquisition._optimize_models()
        self.acquisition._setup()
        self.assertTrue(np.allclose(self.acquisition.fmin.value, 0, atol=1e-1), msg="fmin not updated")

    def test_PoI_validity(self):
        Xcenter = np.random.rand(20, 2) * 0.25 - 0.125
        X = np.random.rand(100, 2) * 2 - 1
        hor_idx = np.abs(X[:, 0]) > 0.8
        ver_idx = np.abs(X[:, 1]) > 0.8
        Xborder = np.vstack((X[hor_idx, :], X[ver_idx, :]))
        poi1 = self.acquisition.evaluate(Xborder)
        poi2 = self.acquisition.evaluate(Xcenter)
        self.assertGreater(np.min(poi2), np.max(poi1))
        self.assertTrue(np.all(self.acquisition.feasible_data_index()), msg="EI does never invalidate points")


class TestProbabilityOfFeasibility(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_plane_model(self.domain)
        self.acquisition = gpflowopt.acquisition.ProbabilityOfFeasibility(self.model)

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

    _multiprocess_can_split_ = True

    def setUp(self):
        self.domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_plane_model(self.domain)
        self.acquisition = gpflowopt.acquisition.LowerConfidenceBound(self.model, 3.2)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="LCB returns all objectives")

    def test_object_integrity(self):
        self.assertEqual(self.acquisition.sigma, 3.2)

    def test_LCB_validity(self):
        design = gpflowopt.design.RandomDesign(200, self.domain).generate()
        q = self.acquisition.evaluate(design)
        p = self.acquisition.models[0].predict_f(design)[0]
        np.testing.assert_array_less(q, p)

    def test_LCB_validity_2(self):
        design = gpflowopt.design.RandomDesign(200, self.domain).generate()
        self.acquisition.sigma = 0
        q = self.acquisition.evaluate(design)
        p = self.acquisition.models[0].predict_f(design)[0]
        np.testing.assert_allclose(q, p)


class TestHVProbabilityOfImprovement(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.model = create_vlmop2_model()
        self.data = load_data('vlmop.npz')
        self.acquisition = gpflowopt.acquisition.HVProbabilityOfImprovement(self.model)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 2, msg="Model list has incorrect length.")
        for m1, m2 in zip(self.acquisition.models, self.model):
            self.assertEqual(m1, m2, msg="Incorrect model stored in ExpectedImprovement")

    def test_HvPoI_validity(self):
        scores = self.acquisition.evaluate(self.data['candidates'])
        np.testing.assert_almost_equal(scores, self.data['scores'], decimal=2)


class TestMinValueEntropySearch(unittest.TestCase):
    def setUp(self):
        super(TestMinValueEntropySearch, self).setUp()
        self.model = self.create_parabola_model()
        self.acquisition = gpflowopt.acquisition.MinValueEntropySearch(self.model, self.domain)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="MinValueEntropySearch returns all objectives")

    def test_setup(self):
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(fmin, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(self.acquisition.samples.shape == (self.acquisition.num_samples,),
                        msg="fmin computed incorrectly")

    def test_MES_validity(self):
        Xcenter = np.random.rand(20, 2) * 0.25 - 0.125
        X = np.random.rand(100, 2) * 2 - 1
        hor_idx = np.abs(X[:, 0]) > 0.8
        ver_idx = np.abs(X[:, 1]) > 0.8
        Xborder = np.vstack((X[hor_idx, :], X[ver_idx, :]))
        ei1 = self.acquisition.evaluate(Xborder)
        ei2 = self.acquisition.evaluate(Xcenter)
        self.assertGreater(np.min(ei2) + 1E-6, np.max(ei1))
        self.assertTrue(np.all(self.acquisition.feasible_data_index()), msg="MES does never invalidate points")
