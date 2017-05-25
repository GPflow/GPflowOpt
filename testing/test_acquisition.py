import GPflowOpt
import unittest
import numpy as np
import GPflow
import tensorflow as tf


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T


def plane(X):
    return X[:, [0]] - 0.5


class _TestAcquisition(object):
    """
    Defines some basic verifications for all acquisition functions. Test classes can derive from this
    """

    @property
    def domain(self):
        return np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])

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

    def setUp(self):
        self.acquisition = None

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
                tf_shape = tens.get_shape().as_list()
                self.assertEqual(tf_shape[0], 50, msg="Tensor of incorrect shape returned")
                self.assertTrue(tf_shape[1] == 1 or tf_shape[1] is None)

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
        np.testing.assert_allclose(self.acquisition.data[0], X, err_msg="Samples not updated")
        np.testing.assert_allclose(self.acquisition.data[1], Y, err_msg="Values not updated")


class TestExpectedImprovement(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestExpectedImprovement, self).setUp()
        self.model = self.create_parabola_model()
        print(self.model)
        self.acquisition = GPflowOpt.acquisition.ExpectedImprovement(self.model)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored in ExpectedImprovement")
        self.assertEqual(len(self.acquisition._default_params), 1)
        print(self.acquisition._default_params[0])
        self.assertTrue(
            np.allclose(np.sort(self.acquisition._default_params[0]), np.sort(np.array([0.5413] * 4)), atol=1e-2),
            msg="Initial hypers improperly stored")
        self.assertEqual(self.acquisition.objective_indices(), np.arange(1, dtype=int),
                         msg="ExpectedImprovement returns all objectives")

    def test_setup(self):
        fmin = np.min(self.acquisition.data[1])
        self.assertGreater(self.acquisition.fmin.value, 0, msg="The minimum (0) is not amongst the design.")
        self.assertTrue(np.allclose(self.acquisition.fmin.value, fmin, atol=1e-2), msg="fmin computed incorrectly")

        # Now add the actual minimum
        p = np.array([[0.0, 0.0]])
        self.acquisition.set_data(np.vstack((self.model.X.value, p)),
                                  np.vstack((self.model.Y.value, parabola2d(p))))
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


class TestProbabilityOfFeasibility(_TestAcquisition, unittest.TestCase):
    def setUp(self):
        super(TestProbabilityOfFeasibility, self).setUp()
        self.model = self.create_plane_model()
        self.acquisition = GPflowOpt.acquisition.ProbabilityOfFeasibility(self.model)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored in PoF")
        self.assertEqual(len(self.acquisition._default_params), 1)
        self.assertTrue(
            np.allclose(np.sort(self.acquisition._default_params[0]), np.sort(np.array([0.5413] * 4)), atol=1e-2),
            msg="Initial hypers improperly stored")
        self.assertEqual(self.acquisition.constraint_indices(), np.arange(1, dtype=int),
                         msg="PoF returns all constraints")

    def test_PoF_validity(self):
        X1 = np.random.rand(10, 2) / 2
        X2 = np.random.rand(10, 2) / 2 + 0.5
        self.assertTrue(np.allclose(self.acquisition.evaluate(X1), 1), msg="Left half of plane is feasible")
        self.assertTrue(np.allclose(self.acquisition.evaluate(X2), 0), msg="Right half of plane is not feasible")


class _TestAcquisitionAggregationOperator(_TestAcquisition):
    def test_object_integrity(self):
        for oper in self.acquisition.operands:
            self.assertTrue(isinstance(oper, GPflowOpt.acquisition.Acquisition),
                            msg="All operands should be an acquisition object")
        self.assertEqual(len(self.acquisition._default_params), 0)

    def test_data(self):
        super(_TestAcquisitionAggregationOperator, self).test_data()
        np.testing.assert_allclose(self.acquisition.data[0], self.acquisition[0].data[0],
                                   err_msg="Samples should be equal for all operands")
        np.testing.assert_allclose(self.acquisition.data[0], self.acquisition[1].data[0],
                                   err_msg="Samples should be equal for all operands")

        Y = np.hstack(map(lambda oper: oper.data[1], self.acquisition.operands))
        np.testing.assert_allclose(self.acquisition.data[1], Y,
                                   err_msg="Value should be horizontally concatenated")


class TestAcquisitionSum(_TestAcquisitionAggregationOperator, unittest.TestCase):
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


class TestAcquisitionProduct(_TestAcquisitionAggregationOperator, unittest.TestCase):
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
        print(p1)
        print(p2)
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


class TestJointAcquisition(unittest.TestCase):
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
        Yc = plane(X)
        m1 = GPflow.gpr.GPR(X, Yo, GPflow.kernels.RBF(2, ARD=True))
        m2 = GPflow.gpr.GPR(X, Yc, GPflow.kernels.RBF(2, ARD=True))
        joint = GPflowOpt.acquisition.ExpectedImprovement(m1) * GPflowOpt.acquisition.ProbabilityOfFeasibility(m2)

        np.testing.assert_allclose(joint.objective_indices(), np.array([0], dtype=int))
        np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

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
