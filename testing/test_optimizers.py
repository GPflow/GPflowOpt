import GPflowOpt
import unittest
import numpy as np
import GPflow


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T, 2 * X


class KeyboardRaiser:
    """
    This wraps a function and makes it raise a KeyboardInterrupt after some number of calls
    """

    def __init__(self, iters_to_raise, f):
        self.iters_to_raise, self.f = iters_to_raise, f
        self.count = 0

    def __call__(self, *a, **kw):
        self.count += 1
        if self.count >= self.iters_to_raise:
            raise KeyboardInterrupt
        return self.f(*a, **kw)


class _TestOptimizer(object):
    def setUp(self):
        self.optimizer = None

    @property
    def domain(self):
        return GPflowOpt.domain.ContinuousParameter("x1", -1.0, 1.0) + \
               GPflowOpt.domain.ContinuousParameter("x2", -1.0, 1.0)

    def test_default_initial(self):
        self.assertTupleEqual(self.optimizer._initial.shape, (1, 2), msg="Invalid shape of initial points array")
        self.assertTrue(np.allclose(self.optimizer._initial, 0), msg="Default initial point incorrect.")

    def test_set_initial(self):
        self.optimizer.set_initial([1, 1])
        self.assertTupleEqual(self.optimizer._initial.shape, (1, 2), msg="Invalid shape of initial points array")
        self.assertTrue(np.allclose(self.optimizer._initial, 1), msg="Specified initial point not loaded.")


class TestCandidateOptimizer(_TestOptimizer, unittest.TestCase):
    def setUp(self):
        super(TestCandidateOptimizer, self).setUp()
        design = GPflowOpt.design.FactorialDesign(4, self.domain)
        self.optimizer = GPflowOpt.optim.CandidateOptimizer(self.domain, design.generate())

    def test_object_integrity(self):
        self.assertTupleEqual(self.optimizer.candidates.shape, (16, 2), msg="Invalid shape of candidate property.")
        self.assertTupleEqual(self.optimizer.get_initial().shape, (17, 2), msg="Invalid shape of initial points")
        self.assertFalse(self.optimizer.gradient_enabled(), msg="CandidateOptimizer supports no gradients.")

    def test_optimize(self):
        result = self.optimizer.optimize(parabola2d)
        self.assertTrue(result.success, msg="Optimization should succeed.")
        self.assertTrue(np.allclose(result.x, 0), msg="Optimum should be identified")
        self.assertTrue(np.allclose(result.fun, 0), msg="Function value in optimum is 0")
        self.assertEqual(result.nfev, 17, msg="Number of function evaluations equals candidates + initial points")

    def test_optimize_second(self):
        self.optimizer.set_initial([0.67, 0.67])
        result = self.optimizer.optimize(parabola2d)
        self.assertGreater(result.fun, 0, msg="Optimum is not amongst candidates and initial points")
        self.assertLess(result.fun, 2, msg="Function value not reachable within domain")


class TestSciPyOptimizer(_TestOptimizer, unittest.TestCase):
    def setUp(self):
        super(TestSciPyOptimizer, self).setUp()
        self.optimizer = GPflowOpt.optim.SciPyOptimizer(self.domain, maxiter=10)

    def test_object_integrity(self):
        self.assertDictEqual(self.optimizer.config, {'tol': None, 'method': 'L-BFGS-B',
                                                     'options': {'maxiter': 10, 'disp': False}},
                             msg="Config dict contains invalid entries.")
        self.assertTrue(self.optimizer.gradient_enabled(), msg="Gradient is supported.")

    def test_optimize(self):
        self.optimizer.set_initial([-1, -1])
        result = self.optimizer.optimize(parabola2d)
        self.assertTrue(result.success)
        self.assertLessEqual(result.nit, 10, "Only 10 Iterations permitted")
        self.assertLessEqual(result.nfev, 20, "Max 20 evaluations permitted")
        self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
        self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")

    def test_optimizer_interrupt(self):
        self.optimizer.set_initial([-1, -1])
        result = self.optimizer.optimize(KeyboardRaiser(2, parabola2d))
        self.assertFalse(result.success, msg="After one evaluation, a keyboard interrupt is raised, "
                                             "non-succesfull result expected.")
        self.assertFalse(np.allclose(result.x, 0), msg="After one iteration, the optimum will not be found")


class TestStagedOptimizer(_TestOptimizer, unittest.TestCase):
    def setUp(self):
        super(TestStagedOptimizer, self).setUp()
        self.optimizer = GPflowOpt.optim.StagedOptimizer([GPflowOpt.optim.MCOptimizer(self.domain, 5),
                                                          GPflowOpt.optim.SciPyOptimizer(self.domain, maxiter=10)])

    def test_object_integrity(self):
        self.assertEqual(len(self.optimizer.optimizers), 2, msg="Two optimizers expected in optimizerlist")
        self.assertFalse(self.optimizer.gradient_enabled(), msg="MCOptimizer supports no gradients => neither "
                                                                "does stagedoptimizer.")

    def test_optimize(self):
        result = self.optimizer.optimize(parabola2d)
        self.assertTrue(result.success)
        self.assertLessEqual(result.nfev, 20, "Only 10 Iterations permitted")
        self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
        self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")

    def test_optimizer_interrupt(self):
        self.optimizer.set_initial([-1, -1])
        result = self.optimizer.optimize(KeyboardRaiser(3, parabola2d))
        self.assertFalse(result.success, msg="After two evaluations, a keyboard interrupt is raised, "
                                             "non-succesfull result expected.")
        self.assertFalse(np.allclose(result.x, 0.0), msg="After one iteration, the optimum will not be found")
        self.assertEqual(result.nstages, 1, msg="Stage 1 should be in progress during interrupt")

        result = self.optimizer.optimize(KeyboardRaiser(8, parabola2d))
        self.assertFalse(result.success, msg="After 7 evaluations, a keyboard interrupt is raised, "
                                             "non-succesfull result expected.")
        self.assertFalse(np.allclose(result.x[0, :], 0.0), msg="The optimum should not be found yet")
        self.assertEqual(result.nstages, 2, msg="Stage 2 should be in progress during interrupt")


class TestBayesianOptimizer(_TestOptimizer, unittest.TestCase):
    def setUp(self):
        super(TestBayesianOptimizer, self).setUp()
        design = GPflowOpt.design.FactorialDesign(4, self.domain)
        X, Y = design.generate(), parabola2d(design.generate())[0]
        model = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True, lengthscales=X.var(axis=0)))
        model.kern.variance.prior = GPflow.priors.Gamma(3, 1 / 3)
        acquisition = GPflowOpt.acquisition.ExpectedImprovement(model)
        self.optimizer = GPflowOpt.BayesianOptimizer(self.domain, acquisition)

    def test_default_initial(self):
        self.assertTupleEqual(self.optimizer._initial.shape, (0, 2), msg="Invalid shape of initial points array")

    def test_optimize(self):
        result = self.optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=20)
        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 20, "Only 20 evaluations permitted")
        self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
        self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")

    def test_optimizer_interrupt(self):
        result = self.optimizer.optimize(KeyboardRaiser(3, lambda X: parabola2d(X)[0]), n_iter=20)
        self.assertFalse(result.success, msg="After 2 evaluations, a keyboard interrupt is raised, "
                                             "non-succesfull result expected.")
        self.assertTrue(np.allclose(result.x, 0.0), msg="The optimum will not be identified nonetheless")

    def test_initial_design(self):
        design = GPflowOpt.design.RandomDesign(5, self.domain)
        optimizer = GPflowOpt.BayesianOptimizer(self.domain, self.optimizer.acquisition, initial=design)

        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=0)
        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 5, "Evaluated only initial")
        self.assertTupleEqual(optimizer.acquisition.data[0].shape, (21, 2))
        self.assertTupleEqual(optimizer.acquisition.data[1].shape, (21, 1))

        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=0)
        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 0, "Initial was not reset")
        self.assertTupleEqual(optimizer.acquisition.data[0].shape, (21, 2))
        self.assertTupleEqual(optimizer.acquisition.data[1].shape, (21, 1))
