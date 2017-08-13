import GPflowOpt
import unittest
import numpy as np
import GPflow
import tensorflow as tf
import os


## CONVENIENT FUNCTIONS ##
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


def load_data(file):
    path = os.path.dirname(os.path.realpath(__file__))
    return np.load(os.path.join(path, 'data', file))


def create_parabola_model(domain, design=None):
    if design is None:
        design = GPflowOpt.design.LatinHyperCube(16, domain)
    X, Y = design.generate(), parabola2d(design.generate())
    m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
    return m


def create_plane_model(domain, design=None):
    if design is None:
        design = GPflowOpt.design.LatinHyperCube(25, domain)
    X, Y = design.generate(), plane(design.generate())
    m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
    return m


def create_vlmop2_model():
    data = load_data('vlmop.npz')
    m1 = GPflow.gpr.GPR(data['X'], data['Y'][:, [0]], kern=GPflow.kernels.Matern32(2))
    m2 = GPflow.gpr.GPR(data['X'], data['Y'][:, [1]], kern=GPflow.kernels.Matern32(2))
    return [m1, m2]


## TESTS ##

class SimpleAcquisition(GPflowOpt.acquisition.Acquisition):
    def __init__(self, model):
        super(SimpleAcquisition, self).__init__(model)
        self.counter = 0

    def setup(self):
        super(SimpleAcquisition, self).setup()
        self.counter += 1

    def build_acquisition(self, Xcand):
        return self.models[0].build_predict(Xcand)[0]


class TestAcquisition(unittest.TestCase):

    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.model = create_parabola_model(self.domain)
        self.acquisition = SimpleAcquisition(self.model)

    def run_setup(self):
        # Optimize models & perform acquisition setup call.
        self.acquisition._optimize_models()
        self.acquisition.setup()

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored.")

    def test_setup_trigger(self):
        m = create_parabola_model(self.domain)
        self.assertTrue(np.allclose(m.get_free_state(), self.acquisition.models[0].get_free_state()))
        self.assertTrue(self.acquisition._needs_setup)
        self.assertEqual(self.acquisition.counter, 0)
        self.acquisition.evaluate(GPflowOpt.design.RandomDesign(10, self.domain).generate())
        self.assertFalse(self.acquisition._needs_setup)
        self.assertEqual(self.acquisition.counter, 1)
        self.assertFalse(np.allclose(m.get_free_state(), self.acquisition.models[0].get_free_state()))

        self.acquisition._needs_setup = True
        self.acquisition.models[0].set_state(m.get_free_state())
        self.acquisition.evaluate_with_gradients(GPflowOpt.design.RandomDesign(10, self.domain).generate())
        self.assertFalse(self.acquisition._needs_setup)
        self.assertEqual(self.acquisition.counter, 2)

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
        # Verify the effect of setting the data
        design = GPflowOpt.design.RandomDesign(10, self.domain)
        X = np.vstack((self.acquisition.data[0], design.generate()))
        Y = parabola2d(X)
        self.acquisition._needs_setup = False
        self.acquisition.set_data(X, Y)
        np.testing.assert_allclose(self.acquisition.data[0], X, atol=1e-5, err_msg="Samples not updated")
        np.testing.assert_allclose(self.acquisition.data[1], Y, atol=1e-5, err_msg="Values not updated")
        self.assertTrue(self.acquisition._needs_setup)

    def test_data_indices(self):
        # Return all data as feasible.
        self.assertTupleEqual(self.acquisition.feasible_data_index().shape, (self.acquisition.data[0].shape[0],))

    def test_enable_scaling(self):
        self.assertFalse(
            any(m.wrapped.X.value in GPflowOpt.domain.UnitCube(self.domain.size) for m in self.acquisition.models))
        self.acquisition._needs_setup = False
        self.acquisition.enable_scaling(self.domain)
        self.assertTrue(
            all(m.wrapped.X.value in GPflowOpt.domain.UnitCube(self.domain.size) for m in self.acquisition.models))
        self.assertTrue(self.acquisition._needs_setup)

    def test_result_shape_tf(self):
        # Verify the returned shape of evaluate
        design = GPflowOpt.design.RandomDesign(50, self.domain)

        with tf.Graph().as_default():
            free_vars = tf.placeholder(tf.float64, [None])
            l = self.acquisition.make_tf_array(free_vars)
            x_tf = tf.placeholder(tf.float64, shape=(50, 2))
            with self.acquisition.tf_mode():
                tens = self.acquisition.build_acquisition(x_tf)
                self.assertTrue(isinstance(tens, tf.Tensor), msg="no Tensor was returned")

    def test_result_shape_np(self):
        design = GPflowOpt.design.RandomDesign(50, self.domain)
        res = self.acquisition.evaluate(design.generate())
        self.assertTupleEqual(res.shape, (50, 1))
        res = self.acquisition.evaluate_with_gradients(design.generate())
        self.assertTrue(isinstance(res, tuple))
        self.assertTrue(len(res), 2)
        self.assertTupleEqual(res[0].shape, (50, 1))
        self.assertTupleEqual(res[1].shape, (50, self.domain.size))


class TestJointAcquisition(unittest.TestCase):

    _multiprocessing_can_split_ = True

    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])

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

        # Test output indices
        np.testing.assert_allclose(joint.objective_indices(), np.array([0], dtype=int))
        np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

        # Test proper setup
        joint._optimize_models()
        joint.setup()
        self.assertGreater(ei.fmin.value, np.min(ei.data[1]), msg="The best objective value is in an infeasible area")
        self.assertTrue(np.allclose(ei.fmin.value, np.min(ei.data[1][pof.feasible_data_index(), :]), atol=1e-3),
                        msg="fmin computed incorrectly")

    def test_hierarchy(self):
        design = GPflowOpt.design.LatinHyperCube(16, self.domain)
        X = design.generate()
        Yc = plane(X)
        m1 = create_parabola_model(self.domain, design)
        m2 = create_parabola_model(self.domain, design)
        m3 = GPflow.gpr.GPR(X, Yc, GPflow.kernels.RBF(2, ARD=True))
        joint = GPflowOpt.acquisition.ExpectedImprovement(m1) * \
                (GPflowOpt.acquisition.ProbabilityOfFeasibility(m3)
                 + GPflowOpt.acquisition.ExpectedImprovement(m2))

        np.testing.assert_allclose(joint.objective_indices(), np.array([0, 2], dtype=int))
        np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

    def test_multi_aggr(self):
        acq = [GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(self.domain)) for i in range(4)]
        acq1, acq2, acq3, acq4 = acq
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
