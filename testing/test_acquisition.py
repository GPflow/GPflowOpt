import gpflowopt
import numpy as np
import gpflow
import tensorflow as tf
from parameterized import parameterized
from .utility import create_parabola_model, parabola2d, plane, GPflowOptTestCase

domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])


class SimpleAcquisition(gpflowopt.acquisition.Acquisition):

    def build_acquisition(self, Xcand):
        return -self.models[0].build_predict(Xcand)[0]


class SimpleParallelBatch(gpflowopt.acquisition.ParallelBatchAcquisition):
    def __init__(self, model):
        super(SimpleParallelBatch, self).__init__(model, batch_size=2)
        self.n_args = 0
        self.counter = 0

    def _setup(self):
        super(SimpleParallelBatch, self)._setup()
        self.counter += 1

    def build_acquisition(self, *args):
        self.n_args = len(args)
        return -tf.add(self.models[0].build_predict(args[0])[0], self.models[0].build_predict(args[1]+1)[0])


class TestParallelBatchAcquisition(GPflowOptTestCase):

    def setUp(self):
        self.model = create_parabola_model(domain)
        self.acquisition = SimpleParallelBatch(self.model)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored.")

    def test_setup_trigger(self):
        with self.test_session():
            Xrand = np.hstack((gpflowopt.design.RandomDesign(10, domain).generate(),
                               gpflowopt.design.RandomDesign(10, domain).generate()))
            m = create_parabola_model(domain)
            self.assertTrue(np.allclose(m.get_free_state(), self.acquisition.models[0].get_free_state()))
            self.assertTrue(self.acquisition._needs_setup)
            self.assertEqual(self.acquisition.counter, 0)
            self.acquisition.evaluate(Xrand)
            self.assertFalse(self.acquisition._needs_setup)
            self.assertEqual(self.acquisition.counter, 1)
            self.assertFalse(np.allclose(m.get_free_state(), self.acquisition.models[0].get_free_state()))

            self.acquisition._needs_setup = True
            self.acquisition.models[0].set_state(m.get_free_state())
            self.acquisition.evaluate_with_gradients(Xrand)
            self.assertFalse(self.acquisition._needs_setup)
            self.assertEqual(self.acquisition.counter, 2)

    def test_data(self):
        # Test the data property
        with self.test_session(graph=tf.Graph()):
            free_vars = tf.placeholder(tf.float64, [None])
            l = self.acquisition.make_tf_array(free_vars)
            with self.acquisition.tf_mode():
                self.assertTrue(isinstance(self.acquisition.data[0], tf.Tensor),
                                msg="data property should return Tensors")
                self.assertTrue(isinstance(self.acquisition.data[1], tf.Tensor),
                                msg="data property should return Tensors")

    def test_data_update(self):
        # Verify the effect of setting the data
        with self.test_session():
            design = gpflowopt.design.RandomDesign(10, domain)
            X = np.vstack((self.acquisition.data[0], design.generate()))
            Y = parabola2d(X)
            self.acquisition._needs_setup = False
            self.acquisition.set_data(X, Y)
            np.testing.assert_allclose(self.acquisition.data[0], X, atol=1e-5, err_msg="Samples not updated")
            np.testing.assert_allclose(self.acquisition.data[1], Y, atol=1e-5, err_msg="Values not updated")
            self.assertTrue(self.acquisition._needs_setup)

    def test_data_indices(self):
        # Return all data as feasible.
        with self.test_session():
            self.assertTupleEqual(self.acquisition.feasible_data_index().shape, (self.acquisition.data[0].shape[0],))

    def test_enable_scaling(self):
        with self.test_session():
            self.assertFalse(
                any(m.wrapped.X.value in gpflowopt.domain.UnitCube(domain.size) for m in self.acquisition.models))
            self.acquisition._needs_setup = False
            self.acquisition.enable_scaling(domain)
            self.assertTrue(
                all(m.wrapped.X.value in gpflowopt.domain.UnitCube(domain.size) for m in self.acquisition.models))
            self.assertTrue(self.acquisition._needs_setup)

    def test_result_shape_tf(self):
        # Verify the returned shape of evaluate
        with self.test_session(graph=tf.Graph()):
            free_vars = tf.placeholder(tf.float64, [None])
            l = self.acquisition.make_tf_array(free_vars)
            x_tf = tf.placeholder(tf.float64, shape=(50, 2))
            with self.acquisition.tf_mode():
                tens = self.acquisition.build_acquisition(x_tf, x_tf)
                self.assertTrue(isinstance(tens, tf.Tensor), msg="no Tensor was returned")

    def test_result_shape_np(self):
        with self.test_session():
            design = gpflowopt.design.RandomDesign(50, domain)
            candidates = np.hstack((design.generate(), design.generate()))
            res = self.acquisition.evaluate(candidates)
            self.assertTupleEqual(res.shape, (50, 1))
            res = self.acquisition.evaluate_with_gradients(candidates)
            self.assertTrue(isinstance(res, tuple))
            self.assertTrue(len(res), 2)
            self.assertTupleEqual(res[0].shape, (50, 1))
            self.assertTupleEqual(res[1].shape, (50, domain.size * 2))

    def test_optimize(self):
        with self.test_session():
            self.acquisition.optimize_restarts = 0
            state = self.acquisition.get_free_state()
            self.acquisition._optimize_models()
            self.assertTrue(np.allclose(state, self.acquisition.get_free_state()))

            self.acquisition.optimize_restarts = 1
            self.acquisition._optimize_models()
            self.assertFalse(np.allclose(state, self.acquisition.get_free_state()))

    def test_arg_splitting(self):
        self.acquisition.evaluate(np.random.rand(10, 4))
        self.assertEqual(self.acquisition.n_args, 2)

        self.acquisition.n_args = 0
        X = np.random.rand(10, 4)
        A = self.acquisition.evaluate_with_gradients(X)
        self.assertEqual(self.acquisition.n_args, 2)

    def test_get_suggestion(self):
        opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 200),
                                               gpflowopt.optim.SciPyOptimizer(domain)])
        Xnew = self.acquisition.get_suggestion(opt)
        self.assertTupleEqual(Xnew.shape, (2, 2))
        self.assertTrue(np.allclose(Xnew[0, :], 0, atol=1e-4))
        self.assertTrue(np.allclose(Xnew[1, :], -1, atol=1e-4))


class TestAcquisition(GPflowOptTestCase):

    def setUp(self):
        self.model = create_parabola_model(domain)
        self.acquisition = SimpleAcquisition(self.model)

    def test_get_suggestion(self):
        opt = gpflowopt.optim.SciPyOptimizer(domain)
        Xnew = self.acquisition.get_suggestion(opt)
        self.assertTupleEqual(Xnew.shape, (1, 2))
        self.assertTrue(np.allclose(Xnew, 0))


aggregations = list()
aggregations.append(gpflowopt.acquisition.AcquisitionSum([
            gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)),
            gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        ]))
aggregations.append(gpflowopt.acquisition.AcquisitionProduct([
            gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)),
            gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        ]))
aggregations.append(gpflowopt.acquisition.MCMCAcquistion(
    gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)), 5)
)


class TestAcquisitionAggregation(GPflowOptTestCase):

    @parameterized.expand(list(zip(aggregations)))
    def test_object_integrity(self, acquisition):
        acquisition._kill_autoflow()
        with self.test_session():
            for oper in acquisition.operands:
                self.assertTrue(isinstance(oper, gpflowopt.acquisition.Acquisition),
                                msg="All operands should be an acquisition object")

            self.assertTrue(all(isinstance(m, gpflowopt.models.ModelWrapper) for m in acquisition.models))

    @parameterized.expand(list(zip(aggregations)))
    def test_data(self, acquisition):
        acquisition._kill_autoflow()
        with self.test_session():
            np.testing.assert_allclose(acquisition.data[0], acquisition[0].data[0],
                                       err_msg="Samples should be equal for all operands")
            np.testing.assert_allclose(acquisition.data[0], acquisition[1].data[0],
                                       err_msg="Samples should be equal for all operands")

            Y = np.hstack(map(lambda model: model.Y.value, acquisition.models))
            np.testing.assert_allclose(acquisition.data[1], Y, err_msg="Value should be horizontally concatenated")

    @parameterized.expand(list(zip(aggregations)))
    def test_enable_scaling(self, acquisition):
        acquisition._kill_autoflow()
        with self.test_session():
            for oper in acquisition.operands:
                self.assertFalse(any(m.wrapped.X.value in gpflowopt.domain.UnitCube(2) for m in oper.models))
            acquisition.enable_scaling(domain)
            for oper in acquisition.operands:
                self.assertTrue(all(m.wrapped.X.value in gpflowopt.domain.UnitCube(2) for m in oper.models))

    @parameterized.expand(list(zip([aggregations[0]])))
    def test_sum_validity(self, acquisition):
        acquisition._kill_autoflow()
        with self.test_session():
            design = gpflowopt.design.FactorialDesign(4, domain)
            m = create_parabola_model(domain)
            single_ei = gpflowopt.acquisition.ExpectedImprovement(m)
            p1 = acquisition.evaluate(design.generate())
            p2 = single_ei.evaluate(design.generate())
            np.testing.assert_allclose(p2, p1 / 2, rtol=1e-3)

    @parameterized.expand(list(zip([aggregations[1]])))
    def test_product_validity(self, acquisition):
        acquisition._kill_autoflow()
        with self.test_session():
            design = gpflowopt.design.FactorialDesign(4, domain)
            m = create_parabola_model(domain)
            single_ei = gpflowopt.acquisition.ExpectedImprovement(m)
            p1 = acquisition.evaluate(design.generate())
            p2 = single_ei.evaluate(design.generate())
            np.testing.assert_allclose(p2, np.sqrt(p1), rtol=1e-3)

    @parameterized.expand(list(zip(aggregations[0:2])))
    def test_indices(self, acquisition):
        acquisition._kill_autoflow()
        np.testing.assert_allclose(acquisition.objective_indices(), np.arange(2, dtype=int))
        np.testing.assert_allclose(acquisition.constraint_indices(), np.arange(0, dtype=int))

    def test_generating_operators(self):
        joint = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)) + \
                gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        self.assertTrue(isinstance(joint, gpflowopt.acquisition.AcquisitionSum))

        joint = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)) * \
                gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain))
        self.assertTrue(isinstance(joint, gpflowopt.acquisition.AcquisitionProduct))

    @parameterized.expand(list(zip([aggregations[2]])))
    def test_hyper_updates(self, acquisition):
        acquisition._kill_autoflow()
        with self.test_session():
            orig_hypers = [c.get_free_state() for c in acquisition.operands[1:]]
            lik_start = acquisition.operands[0].models[0].compute_log_likelihood()
            acquisition._optimize_models()
            self.assertGreater(acquisition.operands[0].models[0].compute_log_likelihood(), lik_start)

            for co, cn in zip(orig_hypers, [c.get_free_state() for c in acquisition.operands[1:]]):
                self.assertFalse(np.allclose(co, cn))

    @parameterized.expand(list(zip([aggregations[2]])))
    def test_marginalized_score(self, acquisition):
        for m in acquisition.models:
            m._needs_recompile = True

        with self.test_session():
            Xt = np.random.rand(20, 2) * 2 - 1
            ei_mle = acquisition.operands[0].evaluate(Xt)
            ei_mcmc = acquisition.evaluate(Xt)
            np.testing.assert_almost_equal(ei_mle, ei_mcmc, decimal=5)

    def test_mcmc_acq(self):
        acquisition = gpflowopt.acquisition.MCMCAcquistion(
            gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)), 10)
        for oper in acquisition.operands:
            self.assertListEqual(acquisition.models, oper.models)
            self.assertEqual(acquisition.operands[0], oper)
        self.assertTrue(acquisition._needs_new_copies)
        acquisition._optimize_models()
        self.assertListEqual(acquisition.models, acquisition.operands[0].models)
        for oper in acquisition.operands[1:]:
            self.assertNotEqual(acquisition.operands[0], oper)
        self.assertFalse(acquisition._needs_new_copies)
        acquisition._setup()
        Xt = np.random.rand(20, 2) * 2 - 1
        ei_mle = acquisition.operands[0].evaluate(Xt)
        ei_mcmc = acquisition.evaluate(Xt)
        np.testing.assert_almost_equal(ei_mle, ei_mcmc, decimal=5)


class TestJointAcquisition(GPflowOptTestCase):

    def test_constrained_ei(self):
        with self.test_session():
            design = gpflowopt.design.LatinHyperCube(16, domain)
            X = design.generate()
            Yo = parabola2d(X)
            Yc = -parabola2d(X) + 0.5
            m1 = gpflow.gpr.GPR(X, Yo, gpflow.kernels.RBF(2, ARD=True, lengthscales=X.std(axis=0)))
            m2 = gpflow.gpr.GPR(X, Yc, gpflow.kernels.RBF(2, ARD=True, lengthscales=X.std(axis=0)))
            ei = gpflowopt.acquisition.ExpectedImprovement(m1)
            pof = gpflowopt.acquisition.ProbabilityOfFeasibility(m2)
            joint = ei * pof

            # Test output indices
            np.testing.assert_allclose(joint.objective_indices(), np.array([0], dtype=int))
            np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

            # Test proper setup
            joint._optimize_models()
            joint._setup()
            self.assertGreater(ei.fmin.value, np.min(ei.data[1]), msg="The best objective value is in an infeasible area")
            self.assertTrue(np.allclose(ei.fmin.value, np.min(ei.data[1][pof.feasible_data_index(), :]), atol=1e-3),
                            msg="fmin computed incorrectly")

    def test_hierarchy(self):
        with self.test_session():
            design = gpflowopt.design.LatinHyperCube(16, domain)
            X = design.generate()
            Yc = plane(X)
            m1 = create_parabola_model(domain, design)
            m2 = create_parabola_model(domain, design)
            m3 = gpflow.gpr.GPR(X, Yc, gpflow.kernels.RBF(2, ARD=True))
            joint = gpflowopt.acquisition.ExpectedImprovement(m1) * \
                    (gpflowopt.acquisition.ProbabilityOfFeasibility(m3)
                     + gpflowopt.acquisition.ExpectedImprovement(m2))

            np.testing.assert_allclose(joint.objective_indices(), np.array([0, 2], dtype=int))
            np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

    def test_multi_aggr(self):
        with self.test_session():
            acq = [gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(domain)) for i in range(4)]
            acq1, acq2, acq3, acq4 = acq
            joint = acq1 + acq2 + acq3
            self.assertIsInstance(joint, gpflowopt.acquisition.AcquisitionSum)
            self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

            joint = acq1 * acq2 * acq3
            self.assertIsInstance(joint, gpflowopt.acquisition.AcquisitionProduct)
            self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

            first = acq2 + acq3
            self.assertIsInstance(first, gpflowopt.acquisition.AcquisitionSum)
            self.assertListEqual(first.operands.sorted_params, [acq2, acq3])
            joint = acq1 + first
            self.assertIsInstance(joint, gpflowopt.acquisition.AcquisitionSum)
            self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

            first = acq2 * acq3
            self.assertIsInstance(first, gpflowopt.acquisition.AcquisitionProduct)
            self.assertListEqual(first.operands.sorted_params, [acq2, acq3])
            joint = acq1 * first
            self.assertIsInstance(joint, gpflowopt.acquisition.AcquisitionProduct)
            self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3])

            first = acq1 + acq2
            second = acq3 + acq4
            joint = first + second
            self.assertIsInstance(joint, gpflowopt.acquisition.AcquisitionSum)
            self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3, acq4])

            first = acq1 * acq2
            second = acq3 * acq4
            joint = first * second
            self.assertIsInstance(joint, gpflowopt.acquisition.AcquisitionProduct)
            self.assertListEqual(joint.operands.sorted_params, [acq1, acq2, acq3, acq4])


class TestRecompile(GPflowOptTestCase):
    """
    Regression test for #37
    """
    def test_vgp(self):
        with self.test_session():
            domain = gpflowopt.domain.UnitCube(2)
            X = gpflowopt.design.RandomDesign(10, domain).generate()
            Y = np.sin(X[:,[0]])
            m = gpflow.vgp.VGP(X, Y, gpflow.kernels.RBF(2), gpflow.likelihoods.Gaussian())
            acq = gpflowopt.acquisition.ExpectedImprovement(m)
            m._compile()
            self.assertFalse(m._needs_recompile)
            acq.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())
            self.assertTrue(hasattr(acq, '_evaluate_AF_storage'))

            Xnew = gpflowopt.design.RandomDesign(5, domain).generate()
            Ynew = np.sin(Xnew[:,[0]])
            acq.set_data(np.vstack((X, Xnew)), np.vstack((Y, Ynew)))
            self.assertFalse(hasattr(acq, '_needs_recompile'))
            self.assertFalse(hasattr(acq, '_evaluate_AF_storage'))
            acq.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())
