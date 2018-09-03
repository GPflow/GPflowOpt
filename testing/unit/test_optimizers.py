import gpflowopt
import numpy as np
import pytest
import gpflow
import six
import sys
import os
import warnings
from contextlib import contextmanager
from scipy.optimize import OptimizeResult
from ..utility import vlmop2, create_parabola_model, create_vlmop2_model, GPflowOptTestCase


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T, 2 * X


class KeyboardRaiser:
    """
    This wraps a function and makes it raise a KeyboardInterrupt after some number of calls
    """

    def __init__(self, iters_to_raise, f):
        self.iters_to_raise, self.f = iters_to_raise, f
        self.count = 0

    def __call__(self, X):
        if self.count >= self.iters_to_raise:
            raise KeyboardInterrupt
        val = self.f(X)
        self.count += X.shape[0]
        return val


class _TestOptimizer(object):

    def setUp(self):
        self.optimizer = None
        warnings.simplefilter("once", category=UserWarning)

    @property
    def domain(self):
        return gpflowopt.domain.ContinuousParameter("x1", -1.0, 1.0) + \
               gpflowopt.domain.ContinuousParameter("x2", -1.0, 1.0)

    def test_default_initial(self):
        self.assertTupleEqual(self.optimizer._initial.shape, (1, 2), msg="Invalid shape of initial points array")
        self.assertTrue(np.allclose(self.optimizer._initial, 0), msg="Default initial point incorrect.")

    def test_set_initial(self):
        self.optimizer.set_initial([1, 1])
        self.assertTupleEqual(self.optimizer._initial.shape, (1, 2), msg="Invalid shape of initial points array")
        self.assertTrue(np.allclose(self.optimizer._initial, 1), msg="Specified initial point not loaded.")

    def test_set_domain(self):
        self.optimizer.domain = gpflowopt.domain.UnitCube(3)
        self.assertNotEqual(self.optimizer.domain, self.domain)
        self.assertEqual(self.optimizer.domain, gpflowopt.domain.UnitCube(3))
        self.assertTrue(np.allclose(self.optimizer.get_initial(), 0.5))


class TestCandidateOptimizer(_TestOptimizer, GPflowOptTestCase):
    def setUp(self):
        super(TestCandidateOptimizer, self).setUp()
        design = gpflowopt.design.FactorialDesign(4, self.domain)
        self.optimizer = gpflowopt.optim.CandidateOptimizer(self.domain, design.generate())

    def test_default_initial(self):
        self.assertTupleEqual(self.optimizer._initial.shape, (0, 2), msg="Invalid shape of initial points array")

    def test_set_initial(self):
        # When run separately this test works, however when calling nose to run all tests on python 2.7 this records
        # no warnings
        with warnings.catch_warnings(record=True) as w:
            self.optimizer.set_initial([1, 1])
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)

    def test_object_integrity(self):
        self.assertTupleEqual(self.optimizer.candidates.shape, (16, 2), msg="Invalid shape of candidate property.")
        self.assertTupleEqual(self.optimizer._get_eval_points().shape, (16, 2))
        self.assertTupleEqual(self.optimizer.get_initial().shape, (0, 2), msg="Invalid shape of initial points")
        self.assertFalse(self.optimizer.gradient_enabled(), msg="CandidateOptimizer supports no gradients.")

    def test_set_domain(self):
        with self.assertRaises(AssertionError):
            super(TestCandidateOptimizer, self).test_set_domain()
        self.optimizer.domain = gpflowopt.domain.UnitCube(2)
        self.assertNotEqual(self.optimizer.domain, self.domain)
        self.assertEqual(self.optimizer.domain, gpflowopt.domain.UnitCube(2))
        rescaled_candidates = gpflowopt.design.FactorialDesign(4, gpflowopt.domain.UnitCube(2)).generate()
        self.assertTrue(np.allclose(self.optimizer.candidates, rescaled_candidates))

    def test_optimize(self):
        self.optimizer.candidates = np.vstack((self.optimizer.candidates, np.zeros((1,2))))
        result = self.optimizer.optimize(parabola2d)
        self.assertTrue(result.success, msg="Optimization should succeed.")
        self.assertTrue(np.allclose(result.x, 0), msg="Optimum should be identified")
        self.assertTrue(np.allclose(result.fun, 0), msg="Function value in optimum is 0")
        self.assertEqual(result.nfev, 17, msg="Number of function evaluations equals candidates + initial points")

    def test_optimize_second(self):
        result = self.optimizer.optimize(parabola2d)
        self.assertGreater(result.fun, 0, msg="Optimum is not amongst candidates and initial points")
        self.assertLess(result.fun, 2, msg="Function value not reachable within domain")


class TestSciPyOptimizer(_TestOptimizer, GPflowOptTestCase):
    def setUp(self):
        super(TestSciPyOptimizer, self).setUp()
        self.optimizer = gpflowopt.optim.SciPyOptimizer(self.domain, maxiter=10)

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


class TestStagedOptimizer(_TestOptimizer, GPflowOptTestCase):
    def setUp(self):
        super(TestStagedOptimizer, self).setUp()
        self.optimizer = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(self.domain, 5),
                                                          gpflowopt.optim.MCOptimizer(self.domain, 5),
                                                          gpflowopt.optim.SciPyOptimizer(self.domain, maxiter=10)])

    def test_default_initial(self):
        self.assertTupleEqual(self.optimizer.optimizers[0]._initial.shape, (0,2))

    def test_set_initial(self):
        self.optimizer.set_initial([1, 1])
        self.assertTupleEqual(self.optimizer.optimizers[0]._initial.shape, (0, 2))
        self.assertTupleEqual(self.optimizer.optimizers[1]._initial.shape, (0, 2))
        self.assertTupleEqual(self.optimizer.optimizers[2]._initial.shape, (1, 2))
        self.assertTupleEqual(self.optimizer.get_initial().shape, (0, 2))

    def test_object_integrity(self):
        self.assertEqual(len(self.optimizer.optimizers), 3, msg="Two optimizers expected in optimizerlist")
        self.assertFalse(self.optimizer.gradient_enabled(), msg="MCOptimizer supports no gradients => neither "
                                                                "does stagedoptimizer.")

    def test_optimize(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = self.optimizer.optimize(parabola2d)
            self.assertTrue(result.success)
            self.assertLessEqual(result.nfev, 20, "Only 20 Iterations permitted")
            self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
            self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")

    def test_optimizer_interrupt(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = self.optimizer.optimize(KeyboardRaiser(0, parabola2d))
            self.assertFalse(result.success, msg="non-succesfull result expected.")
            self.assertEqual(result.nstages, 1, msg="Stage 2 should be in progress during interrupt")
            self.assertEqual(result.nfev, 0)

            result = self.optimizer.optimize(KeyboardRaiser(3, parabola2d))
            self.assertFalse(result.success, msg="non-succesfull result expected.")
            self.assertFalse(np.allclose(result.x, 0.0), msg="The optimum will not be found")
            self.assertEqual(result.nstages, 2, msg="Stage 2 should be in progress during interrupt")
            self.assertEqual(result.nfev, 5)

            result = self.optimizer.optimize(KeyboardRaiser(12, parabola2d))
            self.assertFalse(result.success, msg="non-succesfull result expected.")
            self.assertEqual(result.nfev, 12)
            self.assertFalse(np.allclose(result.x[0, :], 0.0), msg="The optimum should not be found yet")
            self.assertEqual(result.nstages, 3, msg="Stage 3 should be in progress during interrupt")

    def test_set_domain(self):
        super(TestStagedOptimizer, self).test_set_domain()
        for opt in self.optimizer.optimizers:
            self.assertEqual(opt.domain, gpflowopt.domain.UnitCube(3))


class TestBayesianOptimizer(_TestOptimizer, GPflowOptTestCase):
    def setUp(self):
        super(TestBayesianOptimizer, self).setUp()

        acquisition = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(self.domain))
        self.optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition)

    def test_default_initial(self):
        self.assertTupleEqual(self.optimizer._initial.shape, (0, 2), msg="Invalid shape of initial points array")

    def test_optimize(self):
        for verbose in [False, True]:
            with self.test_session():
                acquisition = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(self.domain))
                optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition, verbose=verbose)
                result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=20)
                self.assertTrue(result.success)
                self.assertEqual(result.nfev, 20, "Only 20 evaluations permitted")
                self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
                self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")

    def test_optimize_multi_objective(self):
        for verbose in [False, True]:
            with self.test_session():
                m1, m2 = create_vlmop2_model()
                acquisition = gpflowopt.acquisition.ExpectedImprovement(m1) + gpflowopt.acquisition.ExpectedImprovement(m2)
                optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition, verbose=verbose)
                result = optimizer.optimize(vlmop2, n_iter=2)
                self.assertTrue(result.success)
                self.assertEqual(result.nfev, 2, "Only 2 evaluations permitted")
                self.assertTupleEqual(result.x.shape, (7, 2))
                self.assertTupleEqual(result.fun.shape, (7, 2))
                _, dom = gpflowopt.pareto.non_dominated_sort(result.fun)
                self.assertTrue(np.all(dom == 0))

    def test_optimize_constraint(self):
        for verbose in [False, True]:
            with self.test_session():
                acquisition = gpflowopt.acquisition.ProbabilityOfFeasibility(create_parabola_model(self.domain), threshold=-1)
                optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition, verbose=verbose)
                result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
                self.assertFalse(result.success)
                self.assertEqual(result.message, 'No evaluations satisfied all the constraints')
                self.assertEqual(result.nfev, 1, "Only 1 evaluations permitted")
                self.assertTupleEqual(result.x.shape, (17, 2))
                self.assertTupleEqual(result.fun.shape, (17, 0))
                self.assertTupleEqual(result.constraints.shape, (17, 1))

                acquisition = gpflowopt.acquisition.ProbabilityOfFeasibility(create_parabola_model(self.domain), threshold=0.3)
                optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition, verbose=verbose)
                result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
                self.assertTrue(result.success)
                self.assertEqual(result.nfev, 1, "Only 1 evaluation permitted")
                self.assertTupleEqual(result.x.shape, (5, 2))
                self.assertTupleEqual(result.fun.shape, (5, 0))
                self.assertTupleEqual(result.constraints.shape, (5, 1))

    def test_optimizer_interrupt(self):
        with self.test_session():
            result = self.optimizer.optimize(KeyboardRaiser(3, lambda X: parabola2d(X)[0]), n_iter=20)
            self.assertFalse(result.success, msg="After 2 evaluations, a keyboard interrupt is raised, "
                                                 "failed result expected.")
            self.assertTrue(np.allclose(result.x, 0.0), msg="The optimum will not be identified nonetheless")

    def test_failsafe(self):
        with self.test_session():
            X, Y = self.optimizer.acquisition.data[0], self.optimizer.acquisition.data[1]
            # Provoke cholesky faillure
            self.optimizer.acquisition.optimize_restarts = 1
            self.optimizer.acquisition.models[0].likelihood.variance.transform = gpflow.transforms.Identity()
            self.optimizer.acquisition.models[0].likelihood.variance = -5.0
            self.optimizer.acquisition.models[0]._needs_recompile = True
            with self.assertRaises(RuntimeError) as e:
                with self.optimizer.failsafe():
                    self.optimizer.acquisition.set_data(X, Y)
                    self.optimizer.acquisition.evaluate(X)

            fname = 'failed_bopt_{0}.npz'.format(id(e.exception))
            self.assertTrue(os.path.isfile(fname))
            with np.load(fname) as data:
                np.testing.assert_almost_equal(data['X'], X)
                np.testing.assert_almost_equal(data['Y'], Y)
            os.remove(fname)

    def test_set_domain(self):
        with self.test_session():
            with self.assertRaises(AssertionError):
                super(TestBayesianOptimizer, self).test_set_domain()

            domain = gpflowopt.domain.ContinuousParameter("x1", -2.0, 2.0) + \
                     gpflowopt.domain.ContinuousParameter("x2", -2.0, 2.0)
            self.optimizer.domain = domain
            expected = gpflowopt.design.LatinHyperCube(16, self.domain).generate() / 4 + 0.5
            self.assertTrue(np.allclose(expected, self.optimizer.acquisition.models[0].wrapped.X.value))


class TestBayesianOptimizerConfigurations(GPflowOptTestCase):
    def setUp(self):
        self.domain = gpflowopt.domain.ContinuousParameter("x1", 0.0, 1.0) + \
                      gpflowopt.domain.ContinuousParameter("x2", 0.0, 1.0)
        self.acquisition = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(self.domain))

    def test_initial_design(self):
        with self.test_session():
            design = gpflowopt.design.RandomDesign(5, self.domain)
            optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, initial=design)

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

    def test_mcmc(self):
        with self.test_session():
            optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, hyper_draws=10)
            self.assertIsInstance(optimizer.acquisition, gpflowopt.acquisition.MCMCAcquistion)
            self.assertEqual(len(optimizer.acquisition.operands), 10)
            self.assertEqual(optimizer.acquisition.operands[0], self.acquisition)

            result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=20)
            self.assertTrue(result.success)
            self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
            self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")

    def test_callback(self):
        class DummyCallback(object):
            def __init__(self):
                self.counter = 0

            def __call__(self, models):
                self.counter += 1

        c = DummyCallback()
        optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, callback=c)
        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=2)
        self.assertEqual(c.counter, 2)

    def test_callback_recompile(self):
        class DummyCallback(object):
            def __init__(self):
                self.recompile = False

            def __call__(self, models):
                c = np.random.randint(2, 10)
                models[0].kern.variance.prior = gpflow.priors.Gamma(c, 1./c)
                self.recompile = models[0]._needs_recompile

        c = DummyCallback()
        optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, callback=c)
        self.acquisition.evaluate(np.zeros((1,2))) # Make sure its run and setup to skip
        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
        self.assertFalse(c.recompile)
        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
        self.assertTrue(c.recompile)
        self.assertFalse(self.acquisition.models[0]._needs_recompile)

    def test_callback_recompile_mcmc(self):
        class DummyCallback(object):
            def __init__(self):
                self.no_models = 0

            def __call__(self, models):
                c = np.random.randint(2, 10)
                models[0].kern.variance.prior = gpflow.priors.Gamma(c, 1. / c)
                self.no_models = len(models)

        c = DummyCallback()
        optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, hyper_draws=5, callback=c)
        opers = optimizer.acquisition.operands
        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
        self.assertEqual(c.no_models, 1)
        self.assertEqual(id(opers[0]), id(optimizer.acquisition.operands[0]))
        for op1, op2 in zip(opers[1:], optimizer.acquisition.operands[1:]):
            self.assertNotEqual(id(op1), id(op2))
        opers = optimizer.acquisition.operands
        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
        self.assertEqual(id(opers[0]), id(optimizer.acquisition.operands[0]))
        for op1, op2 in zip(opers[1:], optimizer.acquisition.operands[1:]):
            self.assertNotEqual(id(op1), id(op2))

    def test_nongpr_model(self):
        design = gpflowopt.design.LatinHyperCube(16, self.domain)
        X, Y = design.generate(), parabola2d(design.generate())[0]
        m = gpflow.vgp.VGP(X, Y, gpflow.kernels.RBF(2, ARD=True), likelihood=gpflow.likelihoods.Gaussian())
        acq = gpflowopt.acquisition.ExpectedImprovement(m)
        optimizer = gpflowopt.BayesianOptimizer(self.domain, acq)
        result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
        self.assertTrue(result.success)


class TestSilentOptimization(GPflowOptTestCase):
    @contextmanager
    def captured_output(self):
        # Captures all stdout/stderr
        new_out, new_err = six.StringIO(), six.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def test_silent(self):
        class EmittingOptimizer(gpflowopt.optim.Optimizer):
            def __init__(self):
                super(EmittingOptimizer, self).__init__(gpflowopt.domain.ContinuousParameter('x0', 0, 1))

            def _optimize(self, objective):
                print('hello world!')
                return OptimizeResult(x=np.array([0.5]))

        # First, optimize with silent mode off. Should return the stdout of the optimizer
        opt = EmittingOptimizer()
        with self.captured_output() as (out, err):
            opt.optimize(None)
            output = out.getvalue().strip()
            self.assertEqual(output, 'hello world!')

        # Now with silent mode on
        with self.captured_output() as (out, err):
            with opt.silent():
                opt.optimize(None)
                output = out.getvalue().strip()
                self.assertEqual(output, '')

