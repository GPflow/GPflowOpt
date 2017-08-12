import GPflowOpt
import unittest
import numpy as np
import GPflow
import tensorflow as tf
import os


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T


def plane(X):
    return X[:, [0]] - 0.5


def vlmop2(x):
    transl = 1 / np.sqrt(2)
    part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
    part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
    y1 = 1 - np.exp(-1 * part1)
    y2 = 1 - np.exp(-1 * part2)
    return np.hstack((y1, y2))



class _TestAcquisition(object):
    """
    Defines some basic verifications for all acquisition functions. Test classes can derive from this
    """

    _multiprocess_can_split_ = True

    @property
    def domain(self):
        return np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])

    def load_data(self, file):
        path = os.path.dirname(os.path.realpath(__file__))
        return np.load(os.path.join(path, 'data', file))

    def create_parabola_model(self, design=None):
        if design is None:
            design = GPflowOpt.design.LatinHyperCube(16, self.domain)
        X, Y = design.generate(), parabola2d(design.generate())
        m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
        return m

    def create_plane_model(self, design=None):
        if design is None:
            design = GPflowOpt.design.LatinHyperCube(25, self.domain)
        X, Y = design.generate(), plane(design.generate())
        m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
        return m

    def create_vlmop2_model(self):
        data = self.load_data('vlmop.npz')
        m1 = GPflow.gpr.GPR(data['X'], data['Y'][:, [0]], kern=GPflow.kernels.Matern32(2))
        m2 = GPflow.gpr.GPR(data['X'], data['Y'][:, [1]], kern=GPflow.kernels.Matern32(2))
        return [m1, m2]

    def setUp(self):
        self.acquisition = None
        self.model = None

    def test_result_shape(self):
        # Verify the returned shape of evaluate
        design = GPflowOpt.design.RandomDesign(50, self.domain)

        with tf.Graph().as_default():
            free_vars = tf.placeholder(tf.float64, [None])
            l = self.acquisition.make_tf_array(free_vars)
            x_tf = tf.placeholder(tf.float64, shape=(50, 2))
            with self.acquisition.tf_mode():
                tens = self.acquisition.build_acquisition(x_tf)
                self.assertTrue(isinstance(tens, tf.Tensor), msg="no Tensor was returned")
                # tf_shape = tens.get_shape().as_list()
                # self.assertEqual(tf_shape[0], 50, msg="Tensor of incorrect shape returned")
                # self.assertTrue(tf_shape[1] == 1 or tf_shape[1] is None)

        res = self.acquisition.evaluate(design.generate())
        self.assertTupleEqual(res.shape, (50, 1),
                              msg="Incorrect shape returned for evaluate of {0}".format(self.__class__.__name__))
        res = self.acquisition.evaluate_with_gradients(design.generate())
        self.assertTrue(isinstance(res, tuple))
        self.assertTrue(len(res), 2)
        self.assertTupleEqual(res[0].shape, (50, 1),
                              msg="Incorrect shape returned for evaluate of {0}".format(self.__class__.__name__))
        self.assertTupleEqual(res[1].shape, (50, self.domain.size),
                              msg="Incorrect shape returned for gradient of {0}".format(self.__class__.__name__))

    def test_data(self):
        # Test the data property
        with tf.Graph().as_default():
            free_vars = tf.placeholder(tf.float64, [None])
            l = self.acquisition.make_tf_array(free_vars)
            with self.acquisition.tf_mode():
                self.assertTrue(isinstance(self.acquisition.data[0], tf.Tensor),
                                msg="data property should return Tensors")
                self.assertTrue(isinstance(self.acquisition.data[1], tf.Tensor),
                                msg="data property should return Tensors")

    def test_data_update(self):
        # Verify a data update
        design = GPflowOpt.design.RandomDesign(10, self.domain)
        X = np.vstack((self.acquisition.data[0], design.generate()))
        Y = np.hstack([parabola2d(X)] * self.acquisition.data[1].shape[1])
        self.acquisition.set_data(X, Y)
        np.testing.assert_allclose(self.acquisition.data[0], X, atol=1e-5, err_msg="Samples not updated")
        np.testing.assert_allclose(self.acquisition.data[1], Y, atol=1e-5, err_msg="Values not updated")

    def test_data_indices(self):
        self.assertTupleEqual(self.acquisition.feasible_data_index().shape, (self.acquisition.data[0].shape[0],),
                              msg="Incorrect shape returned.")

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored in ExpectedImprovement")
        self.assertEqual(len(self.acquisition._default_params), 1)
        self.assertTrue(
            np.allclose(np.sort(self.acquisition._default_params[0]), np.sort(np.array([0.5413] * 4)), atol=1e-2),
            msg="Initial hypers improperly stored")

    def test_enable_scaling(self):
        self.assertFalse(
            any(m.wrapped.X.value in GPflowOpt.domain.UnitCube(self.domain.size) for m in self.acquisition.models))
        self.acquisition.enable_scaling(self.domain)
        print(self.acquisition.models[0].wrapped.X.value)
        self.assertTrue(
            all(m.wrapped.X.value in GPflowOpt.domain.UnitCube(self.domain.size) for m in self.acquisition.models))


class TestExpectedImprovement(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestExpectedImprovement, self).setUp()
        self.model = self.create_parabola_model()
        self.acquisition = GPflowOpt.acquisition.ExpectedImprovement(self.model)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="ExpectedImprovement returns all objectives")

    def test_setup(self):
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.acquisition.data[0], p)),
                                  np.vstack((self.acquisition.data[1], parabola2d(p))))
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


class TestProbabilityOfImprovement(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestProbabilityOfImprovement, self).setUp()
        self.model = self.create_parabola_model()
        self.acquisition = GPflowOpt.acquisition.ProbabilityOfImprovement(self.model)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="PoI returns all objectives")

    def test_setup(self):
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.acquisition.data[0], p)),
                                  np.vstack((self.acquisition.data[1], parabola2d(p))))
        self.assertTrue(np.allclose(self.acquisition.fmin.value, 0, atol=1e-1), msg="fmin not updated")


class TestProbabilityOfFeasibility(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestProbabilityOfFeasibility, self).setUp()
        self.model = self.create_plane_model()
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


class TestLowerConfidenceBound(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestLowerConfidenceBound, self).setUp()
        self.model = self.create_plane_model()
        self.acquisition = GPflowOpt.acquisition.LowerConfidenceBound(self.model, 3.2)

    def test_objective_indices(self):
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="LCB returns all objectives")

    def test_object_integrity(self):
        super(TestLowerConfidenceBound, self).test_object_integrity()
        self.assertEqual(self.acquisition.sigma, 3.2)

    def test_LCB_validity(self):
        design = GPflowOpt.design.RandomDesign(200, self.domain).generate()
        p = self.acquisition.models[0].predict_f(design)[0]
        q = self.acquisition.evaluate(design)
        np.testing.assert_array_less(q, p)

    def test_LCB_validity_2(self):
        design = GPflowOpt.design.RandomDesign(200, self.domain).generate()
        self.acquisition.sigma = 0
        p = self.acquisition.models[0].predict_f(design)[0]
        q = self.acquisition.evaluate(design)
        np.testing.assert_allclose(q, p)


class TestHVProbabilityOfImprovement(_TestAcquisition, unittest.TestCase):

    @_TestAcquisition.domain.getter
    def domain(self):
        return np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -2, 2) for i in range(1, 3)])

    def setUp(self):
        super(TestHVProbabilityOfImprovement, self).setUp()
        self.model = self.create_vlmop2_model()
        data = self.load_data('vlmop.npz')
        self.candidates = data['candidates']
        self.acquisition = GPflowOpt.acquisition.HVProbabilityOfImprovement(self.model)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 2, msg="Model list has incorrect length.")
        for m1, m2 in zip(self.acquisition.models, self.model):
            self.assertEqual(m1, m2, msg="Incorrect model stored in ExpectedImprovement")
        self.assertEqual(len(self.acquisition._default_params), 2)
        for i in np.arange(2):
            self.assertTrue(np.allclose(np.sort(self.acquisition._default_params[i]), np.sort(np.array([0.5413] * 3)),
                                        atol=1e-2), msg="Initial hypers improperly stored")

    def test_hvpoi_validity(self):
        scores = self.acquisition.evaluate(self.candidates)
        np.testing.assert_almost_equal(scores.ravel(), np.array(
            [2.23723742e-03, 1.00906739e-03, 1.21152340e-02, 6.51774004e-03, 4.42413300e-03, 3.99320061e-02,
             6.24365778e-04, 1.67279166e-02, 3.70006497e-03, 2.79794264e-02, 1.33966839e-02, 5.08016917e-03,
             7.85351395e-04, 1.69967446e-02, 5.16896760e-03, 3.87581677e-05, 2.59530418e-03, 1.42613142e-02,
             4.71508049e-02, 1.01988869e-02, 4.27149696e-04, 2.20649794e-02, 0.00000000e+00]), decimal=2,
                                       err_msg="hvPoI ranker produced the wrong candidate scores")


class _TestAcquisitionAggregation(_TestAcquisition):
    def test_object_integrity(self):
        for oper in self.acquisition.operands:
            self.assertTrue(isinstance(oper, GPflowOpt.acquisition.Acquisition),
                            msg="All operands should be an acquisition object")
        self.assertEqual(len(self.acquisition._default_params), 0)
        self.assertListEqual(self.acquisition.models.sorted_params, self.models)

    def test_data(self):
        super(_TestAcquisitionAggregation, self).test_data()
        np.testing.assert_allclose(self.acquisition.data[0], self.acquisition[0].data[0],
                                   err_msg="Samples should be equal for all operands")
        np.testing.assert_allclose(self.acquisition.data[0], self.acquisition[1].data[0],
                                   err_msg="Samples should be equal for all operands")

        Y = np.hstack(map(lambda model: model.Y.value, self.acquisition.models))
        np.testing.assert_allclose(self.acquisition.data[1], Y,
                                   err_msg="Value should be horizontally concatenated")

    def test_enable_scaling(self):
        for oper in self.acquisition.operands:
            self.assertFalse(any(m.wrapped.X.value in GPflowOpt.domain.UnitCube(self.domain.size) for m in oper.models))
        self.acquisition.enable_scaling(self.domain)
        for oper in self.acquisition.operands:
            self.assertTrue(all(m.wrapped.X.value in GPflowOpt.domain.UnitCube(self.domain.size) for m in oper.models))


class TestAcquisitionSum(_TestAcquisitionAggregation, unittest.TestCase):
    def setUp(self):
        super(TestAcquisitionSum, self).setUp()
        self.models = [self.create_parabola_model(), self.create_parabola_model()]
        self.acquisition = GPflowOpt.acquisition.AcquisitionSum([
            GPflowOpt.acquisition.ExpectedImprovement(self.models[0]),
            GPflowOpt.acquisition.ExpectedImprovement(self.models[1])
        ])

    def test_sum_validity(self):
        design = GPflowOpt.design.FactorialDesign(4, self.domain)
        m = self.create_parabola_model()
        single_ei = GPflowOpt.acquisition.ExpectedImprovement(m)
        p1 = self.acquisition.evaluate(design.generate())
        p2 = single_ei.evaluate(design.generate())
        np.testing.assert_allclose(p2, p1 / 2, rtol=1e-3, err_msg="The sum of 2 EI should be the double of only EI")

    def test_generating_operator(self):
        joint = GPflowOpt.acquisition.ExpectedImprovement(self.create_parabola_model()) + \
                GPflowOpt.acquisition.ExpectedImprovement(self.create_parabola_model())
        self.assertTrue(isinstance(joint, GPflowOpt.acquisition.AcquisitionSum))

    def test_indices(self):
        np.testing.assert_allclose(self.acquisition.objective_indices(), np.arange(2, dtype=int),
                                   err_msg="Sum of two EI should return all objectives")
        np.testing.assert_allclose(self.acquisition.constraint_indices(), np.arange(0, dtype=int),
                                   err_msg="Sum of two EI should return no constraints")


class TestAcquisitionProduct(_TestAcquisitionAggregation, unittest.TestCase):
    def setUp(self):
        super(TestAcquisitionProduct, self).setUp()
        self.models = [self.create_parabola_model(), self.create_parabola_model()]
        self.acquisition = GPflowOpt.acquisition.AcquisitionProduct([
            GPflowOpt.acquisition.ExpectedImprovement(self.models[0]),
            GPflowOpt.acquisition.ExpectedImprovement(self.models[1])
        ])

    def test_product_validity(self):
        design = GPflowOpt.design.FactorialDesign(4, self.domain)
        m = self.create_parabola_model()
        single_ei = GPflowOpt.acquisition.ExpectedImprovement(m)
        p1 = self.acquisition.evaluate(design.generate())
        p2 = single_ei.evaluate(design.generate())
        np.testing.assert_allclose(p2, np.sqrt(p1), rtol=1e-3,
                                   err_msg="The product of 2 EI should be the square of one EI")

    def test_generating_operator(self):
        joint = GPflowOpt.acquisition.ExpectedImprovement(self.create_parabola_model()) * \
                GPflowOpt.acquisition.ExpectedImprovement(self.create_parabola_model())
        self.assertTrue(isinstance(joint, GPflowOpt.acquisition.AcquisitionProduct))

    def test_indices(self):
        np.testing.assert_allclose(self.acquisition.objective_indices(), np.arange(2, dtype=int),
                                   err_msg="Product of two EI should return all objectives")
        np.testing.assert_allclose(self.acquisition.constraint_indices(), np.arange(0, dtype=int),
                                   err_msg="Product of two EI should return no constraints")


class TestMCMCAcquisition(_TestAcquisitionAggregation, unittest.TestCase):
    def setUp(self):
        super(TestMCMCAcquisition, self).setUp()
        self.models = [self.create_parabola_model()]
        self.acquisition = GPflowOpt.acquisition.MCMCAcquistion(
            GPflowOpt.acquisition.ExpectedImprovement(self.models[0]), 5)

    def test_hyper_updates(self):
        orig_hypers = [c.get_free_state() for c in self.acquisition.operands[1:]]
        self.acquisition._update_hyper_draws()
        for co, cn in zip(orig_hypers, [c.get_free_state() for c in self.acquisition.operands[1:]]):
            self.assertFalse(np.allclose(co, cn))

    def test_marginalized_score(self):
        Xt = np.random.rand(20, 2) * 2 - 1
        ei_mle = self.acquisition.operands[0].evaluate(Xt)
        ei_mcmc = self.acquisition.evaluate(Xt)
        np.testing.assert_almost_equal(ei_mle, ei_mcmc, decimal=5)


class TestJointAcquisition(unittest.TestCase):
    _multiprocessing_can_split_ = True

    @property
    def domain(self):
        return np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])

    def create_parabola_model(self, design=None):
        if design is None:
            design = GPflowOpt.design.LatinHyperCube(16, self.domain)
        X, Y = design.generate(), parabola2d(design.generate())
        m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
        return m

    def test_constrained_EI(self):
        design = GPflowOpt.design.LatinHyperCube(16, self.domain)
        X = design.generate()
        Yo = parabola2d(X)
        Yc = -parabola2d(X) + 0.5
        m1 = GPflow.gpr.GPR(X, Yo, GPflow.kernels.RBF(2, ARD=True, lengthscales=X.std(axis=0)))
        m2 = GPflow.gpr.GPR(X, Yc, GPflow.kernels.RBF(2, ARD=True, lengthscales=X.std(axis=0)))
        ei = GPflowOpt.acquisition.ExpectedImprovement(m1)
        pof = GPflowOpt.acquisition.ProbabilityOfFeasibility(m2)
        joint = ei * pof

        np.testing.assert_allclose(joint.objective_indices(), np.array([0], dtype=int))
        np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))
        self.assertGreater(ei.fmin.value, np.min(ei.data[1]), msg="The best objective value is in an infeasible area")
        self.assertTrue(np.allclose(ei.fmin.value, np.min(ei.data[1][pof.feasible_data_index(), :]), atol=1e-3),
                        msg="fmin computed incorrectly")

    def test_hierarchy(self):
        design = GPflowOpt.design.LatinHyperCube(16, self.domain)
        X = design.generate()
        Yc = plane(X)
        m1 = self.create_parabola_model()
        m2 = self.create_parabola_model()
        m3 = GPflow.gpr.GPR(X, Yc, GPflow.kernels.RBF(2, ARD=True))
        joint = GPflowOpt.acquisition.ExpectedImprovement(m1) * \
                (GPflowOpt.acquisition.ProbabilityOfFeasibility(m3)
                 + GPflowOpt.acquisition.ExpectedImprovement(m2))

        np.testing.assert_allclose(joint.objective_indices(), np.array([0, 2], dtype=int))
        np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

    def test_multi_aggr(self):
        models = [self.create_parabola_model(), self.create_parabola_model(), self.create_parabola_model()]
        acq1, acq2, acq3 = tuple(map(lambda m: GPflowOpt.acquisition.ExpectedImprovement(m), models))
        joint = acq1 + acq2 + acq3
        self.assertIsInstance(joint, GPflowOpt.acquisition.AcquisitionSum)
        self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

        joint = acq1 * acq2 * acq3
        self.assertIsInstance(joint, GPflowOpt.acquisition.AcquisitionProduct)
        self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

        first = acq2 + acq3
        self.assertIsInstance(first, GPflowOpt.acquisition.AcquisitionSum)
        self.assertListEqual(first.operands.sorted_params, [acq2, acq3])
        joint = acq1 + first
        self.assertIsInstance(joint, GPflowOpt.acquisition.AcquisitionSum)
        self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

        first = acq2 * acq3
        self.assertIsInstance(first, GPflowOpt.acquisition.AcquisitionProduct)
        self.assertListEqual(first.operands.sorted_params, [acq2, acq3])
        joint = acq1 * first
        self.assertIsInstance(joint, GPflowOpt.acquisition.AcquisitionProduct)
        self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

        acq4 = GPflowOpt.acquisition.ExpectedImprovement(self.create_parabola_model())
        first = acq1 + acq2
        second = acq3 + acq4
        joint = first + second
        self.assertIsInstance(joint, GPflowOpt.acquisition.AcquisitionSum)
        self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3, acq4])

        first = acq1 * acq2
        second = acq3 * acq4
        joint = first * second
        self.assertIsInstance(joint, GPflowOpt.acquisition.AcquisitionProduct)
        self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3, acq4])


class TestMinValueEntropySearch(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestMinValueEntropySearch, self).setUp()
        self.model = self.create_parabola_model()
        self.acquisition = GPflowOpt.acquisition.MinValueEntropySearch(self.model, self.domain)

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
