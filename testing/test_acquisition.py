import GPflowOpt
import unittest
import numpy as np
import GPflow
import tensorflow as tf
from parameterized import parameterized
from .utility import create_parabola_model, parabola2d, plane

domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])


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

    _multiprocess_can_split_ = True

    def setUp(self):
        self.model = create_parabola_model(domain)
        self.acquisition = SimpleAcquisition(self.model)

    def run_setup(self):
        # Optimize models & perform acquisition setup call.
        self.acquisition._optimize_models()
        self.acquisition.setup()

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored.")

    def test_setup_trigger(self):
        m = create_parabola_model(domain)
        self.assertTrue(np.allclose(m.get_free_state(), self.acquisition.models[0].get_free_state()))
        self.assertTrue(self.acquisition._needs_setup)
        self.assertEqual(self.acquisition.counter, 0)
        self.acquisition.evaluate(GPflowOpt.design.RandomDesign(10, domain).generate())
        self.assertFalse(self.acquisition._needs_setup)
        self.assertEqual(self.acquisition.counter, 1)
        self.assertFalse(np.allclose(m.get_free_state(), self.acquisition.models[0].get_free_state()))

        self.acquisition._needs_setup = True
        self.acquisition.models[0].set_state(m.get_free_state())
        self.acquisition.evaluate_with_gradients(GPflowOpt.design.RandomDesign(10, domain).generate())
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
        design = GPflowOpt.design.RandomDesign(10, domain)
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
            any(m.wrapped.X.value in GPflowOpt.domain.UnitCube(domain.size) for m in self.acquisition.models))
        self.acquisition._needs_setup = False
        self.acquisition.enable_scaling(domain)
        self.assertTrue(
            all(m.wrapped.X.value in GPflowOpt.domain.UnitCube(domain.size) for m in self.acquisition.models))
        self.assertTrue(self.acquisition._needs_setup)

    def test_result_shape_tf(self):
        # Verify the returned shape of evaluate
        design = GPflowOpt.design.RandomDesign(50, domain)

        with tf.Graph().as_default():
            free_vars = tf.placeholder(tf.float64, [None])
            l = self.acquisition.make_tf_array(free_vars)
            x_tf = tf.placeholder(tf.float64, shape=(50, 2))
            with self.acquisition.tf_mode():
                tens = self.acquisition.build_acquisition(x_tf)
                self.assertTrue(isinstance(tens, tf.Tensor), msg="no Tensor was returned")

    def test_result_shape_np(self):
        design = GPflowOpt.design.RandomDesign(50, domain)
        res = self.acquisition.evaluate(design.generate())
        self.assertTupleEqual(res.shape, (50, 1))
        res = self.acquisition.evaluate_with_gradients(design.generate())
        self.assertTrue(isinstance(res, tuple))
        self.assertTrue(len(res), 2)
        self.assertTupleEqual(res[0].shape, (50, 1))
        self.assertTupleEqual(res[1].shape, (50, domain.size))

    def test_optimize(self):
        self.acquisition.optimize_restarts = 0
        state = self.acquisition.get_free_state()
        self.acquisition._optimize_models()
        self.assertTrue(np.allclose(state, self.acquisition.get_free_state()))

        self.acquisition.optimize_restarts = 1
        self.acquisition._optimize_models()
        self.assertFalse(np.allclose(state, self.acquisition.get_free_state()))


aggregations = list()
aggregations.append(GPflowOpt.acquisition.AcquisitionSum([
            GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain)),
            GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        ]))
aggregations.append(GPflowOpt.acquisition.AcquisitionProduct([
            GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain)),
            GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        ]))
aggregations.append(GPflowOpt.acquisition.MCMCAcquistion(
    GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain)), 5)
)


class TestAcquisitionAggregation(unittest.TestCase):

    _multiprocess_can_split_ = True

    @parameterized.expand(list(zip(aggregations)))
    def test_object_integrity(self, acquisition):
        for oper in acquisition.operands:
            self.assertTrue(isinstance(oper, GPflowOpt.acquisition.Acquisition),
                            msg="All operands should be an acquisition object")
            self.assertTrue(all(isinstance(m, GPflowOpt.models.ModelWrapper) for m in acquisition.models))


    @parameterized.expand(list(zip(aggregations)))
    def test_data(self, acquisition):
        np.testing.assert_allclose(acquisition.data[0], acquisition[0].data[0],
                                   err_msg="Samples should be equal for all operands")
        np.testing.assert_allclose(acquisition.data[0], acquisition[1].data[0],
                                   err_msg="Samples should be equal for all operands")

        Y = np.hstack(map(lambda model: model.Y.value, acquisition.models))
        np.testing.assert_allclose(acquisition.data[1], Y, err_msg="Value should be horizontally concatenated")

    @parameterized.expand(list(zip(aggregations)))
    def test_enable_scaling(self, acquisition):
        for oper in acquisition.operands:
            self.assertFalse(any(m.wrapped.X.value in GPflowOpt.domain.UnitCube(2) for m in oper.models))
        acquisition.enable_scaling(domain)
        for oper in acquisition.operands:
            self.assertTrue(all(m.wrapped.X.value in GPflowOpt.domain.UnitCube(2) for m in oper.models))

    @parameterized.expand(list(zip([aggregations[0]])))
    def test_sum_validity(self, acquisition):
        design = GPflowOpt.design.FactorialDesign(4, domain)
        m = create_parabola_model(domain)
        single_ei = GPflowOpt.acquisition.ExpectedImprovement(m)
        p1 = acquisition.evaluate(design.generate())
        p2 = single_ei.evaluate(design.generate())
        np.testing.assert_allclose(p2, p1 / 2, rtol=1e-3)

    @parameterized.expand(list(zip([aggregations[1]])))
    def test_product_validity(self, acquisition):
        design = GPflowOpt.design.FactorialDesign(4, domain)
        m = create_parabola_model(domain)
        single_ei = GPflowOpt.acquisition.ExpectedImprovement(m)
        p1 = acquisition.evaluate(design.generate())
        p2 = single_ei.evaluate(design.generate())
        np.testing.assert_allclose(p2, np.sqrt(p1), rtol=1e-3)

    @parameterized.expand(list(zip(aggregations[0:2])))
    def test_indices(self, acquisition):
        np.testing.assert_allclose(acquisition.objective_indices(), np.arange(2, dtype=int))
        np.testing.assert_allclose(acquisition.constraint_indices(), np.arange(0, dtype=int))

    def test_generating_operators(self):
        joint = GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain)) + \
                GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        self.assertTrue(isinstance(joint, GPflowOpt.acquisition.AcquisitionSum))

        joint = GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain)) * \
                GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        self.assertTrue(isinstance(joint, GPflowOpt.acquisition.AcquisitionProduct))

    @parameterized.expand(list(zip([aggregations[2]])))
    def test_hyper_updates(self, acquisition):
        orig_hypers = [c.get_free_state() for c in acquisition.operands[1:]]
        lik_start = acquisition.operands[0].models[0].compute_log_likelihood()
        acquisition._optimize_models()
        self.assertGreater(acquisition.operands[0].models[0].compute_log_likelihood(), lik_start)

        for co, cn in zip(orig_hypers, [c.get_free_state() for c in acquisition.operands[1:]]):
            self.assertFalse(np.allclose(co, cn))

    @parameterized.expand(list(zip([aggregations[2]])))
    def test_marginalized_score(self, acquisition):
        acquisition._optimize_models()
        acquisition.setup()
        Xt = np.random.rand(20, 2) * 2 - 1
        ei_mle = acquisition.operands[0].evaluate(Xt)
        ei_mcmc = acquisition.evaluate(Xt)
        np.testing.assert_almost_equal(ei_mle, ei_mcmc, decimal=5)

    @parameterized.expand(list(zip([aggregations[2]])))
    def test_mcmc_acq_models(self, acquisition):
        self.assertListEqual(acquisition.models, acquisition.operands[0].models)


class TestJointAcquisition(unittest.TestCase):

    _multiprocessing_can_split_ = True

    def test_constrained_EI(self):
        design = GPflowOpt.design.LatinHyperCube(16, domain)
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
        design = GPflowOpt.design.LatinHyperCube(16, domain)
        X = design.generate()
        Yc = plane(X)
        m1 = create_parabola_model(domain, design)
        m2 = create_parabola_model(domain, design)
        m3 = GPflow.gpr.GPR(X, Yc, GPflow.kernels.RBF(2, ARD=True))
        joint = GPflowOpt.acquisition.ExpectedImprovement(m1) * \
                (GPflowOpt.acquisition.ProbabilityOfFeasibility(m3)
                 + GPflowOpt.acquisition.ExpectedImprovement(m2))

        np.testing.assert_allclose(joint.objective_indices(), np.array([0, 2], dtype=int))
        np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

    def test_multi_aggr(self):
        acq = [GPflowOpt.acquisition.ExpectedImprovement(create_parabola_model(domain)) for i in range(4)]
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


class TestRecompile(unittest.TestCase):
    """
    Regression test for #37
    """
    def test_vgp(self):
        domain = GPflowOpt.domain.UnitCube(2)
        X = GPflowOpt.design.RandomDesign(10, domain).generate()
        Y = np.sin(X[:,[0]])
        m = GPflow.vgp.VGP(X, Y, GPflow.kernels.RBF(2), GPflow.likelihoods.Gaussian())
        m._compile()
        acq = GPflowOpt.acquisition.ExpectedImprovement(m)
        self.assertFalse(m._needs_recompile)
        acq.evaluate(GPflowOpt.design.RandomDesign(10, domain).generate())
        self.assertTrue(hasattr(acq, '_evaluate_AF_storage'))

        Xnew = GPflowOpt.design.RandomDesign(5, domain).generate()
        Ynew = np.sin(Xnew[:,[0]])
        acq.set_data(np.vstack((X, Xnew)), np.vstack((Y, Ynew)))
        self.assertFalse(hasattr(acq, '_needs_recompile'))
        self.assertFalse(hasattr(acq, '_evaluate_AF_storage'))
        acq.evaluate(GPflowOpt.design.RandomDesign(10, domain).generate())
