import gpflowopt
import numpy as np
import warnings
import pytest
from ..utility import parabola2d_grad, KeyboardRaiser


@pytest.fixture(scope="module")
def design(domain):
    yield gpflowopt.design.FactorialDesign(4, domain)


class TestCandidateOptimizer(object):

    @pytest.fixture()
    def optimizer(self, domain, design):
        yield gpflowopt.optim.CandidateOptimizer(domain, design.generate())

    def test_default_initial(self, optimizer):
        assert optimizer._initial.shape == (0, 2)

    def test_set_initial(self, optimizer):
        # When run separately this test works, however when calling nose to run all tests on python 2.7 this records
        # no warnings
        with warnings.catch_warnings(record=True) as w:
            optimizer.set_initial([1, 1])
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)

    def test_set_domain(self, domain, optimizer):
        with pytest.raises(AssertionError):
            optimizer.domain = gpflowopt.domain.UnitCube(3)
        optimizer.domain = gpflowopt.domain.UnitCube(2)
        assert optimizer.domain != domain
        assert optimizer.domain == gpflowopt.domain.UnitCube(2)
        rescaled_candidates = gpflowopt.design.FactorialDesign(4, gpflowopt.domain.UnitCube(2)).generate()
        np.testing.assert_allclose(optimizer.candidates, rescaled_candidates)

    def test_object_integrity(self, optimizer):
        assert optimizer.candidates.shape == (16, 2)
        assert optimizer._get_eval_points().shape == (16, 2)
        assert optimizer.get_initial().shape == (0, 2)
        assert not optimizer.gradient_enabled()

    def test_optimize(self, optimizer):
        optimizer.candidates = np.vstack((optimizer.candidates, np.zeros((1,2))))
        result = optimizer.optimize(parabola2d_grad)
        assert result.success
        np.testing.assert_allclose(result.x, 0)
        np.testing.assert_allclose(result.fun, 0)
        assert result.nfev == 17

    def test_optimize_second(self, optimizer):
        result = optimizer.optimize(parabola2d_grad)
        assert result.fun > 0
        assert result.fun < 2


class TestSciPyOptimizer(object):

    @pytest.fixture()
    def optimizer(self, domain):
        yield gpflowopt.optim.SciPyOptimizer(domain, maxiter=10)

    def test_object_integrity(self, optimizer):
        assert optimizer.config == {'tol': None, 'method': 'L-BFGS-B', 'options': {'maxiter': 10, 'disp': False}}
        assert optimizer.gradient_enabled()

    def test_optimize(self, optimizer):
        optimizer.set_initial([-1, -1])
        result = optimizer.optimize(parabola2d_grad)
        assert result.success
        assert result.nit <= 10
        assert result.nfev <= 20
        np.testing.assert_allclose(result.x, 0)
        np.testing.assert_allclose(result.fun, 0)

    def test_optimizer_interrupt(self, optimizer):
        optimizer.set_initial([-1, -1])
        result = optimizer.optimize(KeyboardRaiser(2, parabola2d_grad))
        assert not result.success
        assert not np.allclose(result.x, 0)

    def test_default_initial(self, optimizer):
        assert optimizer._initial.shape == (1, 2)
        np.testing.assert_allclose(optimizer._initial, 0)

    def test_set_initial(self, optimizer):
        optimizer.set_initial([1, 1])
        assert optimizer._initial.shape == (1, 2)
        np.testing.assert_allclose(optimizer._initial, 1)

    def test_set_domain(self, domain, optimizer):
        optimizer.domain = gpflowopt.domain.UnitCube(3)
        assert optimizer.domain != domain
        assert optimizer.domain == gpflowopt.domain.UnitCube(3)
        np.testing.assert_allclose(optimizer.get_initial(), 0.5)


class TestStagedOptimizer(object):

    @pytest.fixture()
    def optimizer(self, domain):
        yield gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 5),
                                               gpflowopt.optim.MCOptimizer(domain, 5),
                                               gpflowopt.optim.SciPyOptimizer(domain, maxiter=10)])

    def test_default_initial(self, optimizer):
        assert optimizer.optimizers[0]._initial.shape == (0, 2)

    def test_set_initial(self, optimizer):
        optimizer.set_initial([1, 1])
        assert optimizer.optimizers[0]._initial.shape == (0, 2)
        assert optimizer.optimizers[1]._initial.shape == (0, 2)
        assert optimizer.optimizers[2]._initial.shape == (1, 2)
        assert optimizer.get_initial().shape == (0, 2)

    def test_object_integrity(self, optimizer):
        assert len(optimizer.optimizers) == 3
        assert not optimizer.gradient_enabled()

    def test_optimize(self, optimizer):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = optimizer.optimize(parabola2d_grad)
            assert result.success
            assert result.nfev <= 20
            np.testing.assert_allclose(result.x, 0)
            np.testing.assert_allclose(result.fun, 0)

    def test_optimizer_interrupt(self, optimizer):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = optimizer.optimize(KeyboardRaiser(0, parabola2d_grad))
            assert not result.success
            assert result.nstages == 1
            assert result.nfev == 0

            result = optimizer.optimize(KeyboardRaiser(3, parabola2d_grad))
            assert not result.success
            assert not np.allclose(result.x, 0.0)
            assert result.nstages == 2
            assert result.nfev == 5

            result = optimizer.optimize(KeyboardRaiser(12, parabola2d_grad))
            assert not result.success
            assert result.nfev == 12
            assert not np.allclose(result.x[0, :], 0.0)
            assert result.nstages == 3

    def test_set_domain(self, domain, optimizer):
        optimizer.domain = gpflowopt.domain.UnitCube(3)
        assert optimizer.domain != domain
        assert optimizer.domain == gpflowopt.domain.UnitCube(3)
        np.testing.assert_allclose(optimizer.get_initial(), 0.5)
        for opt in optimizer.optimizers:
            assert opt.domain == gpflowopt.domain.UnitCube(3)



# class _TestOptimizer(object):
#
#     def setUp(self):
#         self.optimizer = None
#         warnings.simplefilter("once", category=UserWarning)
#
#     @property
#     def domain(self):
#         return gpflowopt.domain.ContinuousParameter("x1", -1.0, 1.0) + \
#                gpflowopt.domain.ContinuousParameter("x2", -1.0, 1.0)
#
#     def test_default_initial(self):
#         self.assertTupleEqual(self.optimizer._initial.shape, (1, 2), msg="Invalid shape of initial points array")
#         self.assertTrue(np.allclose(self.optimizer._initial, 0), msg="Default initial point incorrect.")
#
#     def test_set_initial(self):
#         self.optimizer.set_initial([1, 1])
#         self.assertTupleEqual(self.optimizer._initial.shape, (1, 2), msg="Invalid shape of initial points array")
#         self.assertTrue(np.allclose(self.optimizer._initial, 1), msg="Specified initial point not loaded.")
#
#     def test_set_domain(self):
#         self.optimizer.domain = gpflowopt.domain.UnitCube(3)
#         self.assertNotEqual(self.optimizer.domain, self.domain)
#         self.assertEqual(self.optimizer.domain, gpflowopt.domain.UnitCube(3))
#         self.assertTrue(np.allclose(self.optimizer.get_initial(), 0.5))
#
#


#
#
# class TestBayesianOptimizer(_TestOptimizer, GPflowOptTestCase):
#     def setUp(self):
#         super(TestBayesianOptimizer, self).setUp()
#         acquisition = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(self.domain))
#         self.optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition)
#
#     def test_default_initial(self):
#         self.assertTupleEqual(self.optimizer._initial.shape, (0, 2), msg="Invalid shape of initial points array")
#
#     def test_optimize(self):
#         with self.test_session():
#             result = self.optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=20)
#             self.assertTrue(result.success)
#             self.assertEqual(result.nfev, 20, "Only 20 evaluations permitted")
#             self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
#             self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")
#
#     def test_optimize_multi_objective(self):
#         with self.test_session():
#             m1, m2 = create_vlmop2_model()
#             acquisition = gpflowopt.acquisition.ExpectedImprovement(m1) + gpflowopt.acquisition.ExpectedImprovement(m2)
#             optimizer = gpflowopt.BayesianOptimizer(self.domain, acquisition)
#             result = optimizer.optimize(vlmop2, n_iter=2)
#             self.assertTrue(result.success)
#             self.assertEqual(result.nfev, 2, "Only 2 evaluations permitted")
#             self.assertTupleEqual(result.x.shape, (7, 2))
#             self.assertTupleEqual(result.fun.shape, (7, 2))
#             _, dom = gpflowopt.pareto.non_dominated_sort(result.fun)
#             self.assertTrue(np.all(dom==0))
#
#     def test_optimizer_interrupt(self):
#         with self.test_session():
#             result = self.optimizer.optimize(KeyboardRaiser(3, lambda X: parabola2d(X)[0]), n_iter=20)
#             self.assertFalse(result.success, msg="After 2 evaluations, a keyboard interrupt is raised, "
#                                                  "non-succesfull result expected.")
#             self.assertTrue(np.allclose(result.x, 0.0), msg="The optimum will not be identified nonetheless")
#
#     def test_failsafe(self):
#         with self.test_session():
#             X, Y = self.optimizer.acquisition.data[0], self.optimizer.acquisition.data[1]
#             # Provoke cholesky faillure
#             self.optimizer.acquisition.optimize_restarts = 1
#             self.optimizer.acquisition.models[0].likelihood.variance.transform = gpflow.transforms.Identity()
#             self.optimizer.acquisition.models[0].likelihood.variance = -5.0
#             self.optimizer.acquisition.models[0]._needs_recompile = True
#             with self.assertRaises(RuntimeError) as e:
#                 with self.optimizer.failsafe():
#                     self.optimizer.acquisition.set_data(X, Y)
#                     self.optimizer.acquisition.evaluate(X)
#
#             fname = 'failed_bopt_{0}.npz'.format(id(e.exception))
#             self.assertTrue(os.path.isfile(fname))
#             with np.load(fname) as data:
#                 np.testing.assert_almost_equal(data['X'], X)
#                 np.testing.assert_almost_equal(data['Y'], Y)
#             os.remove(fname)
#
#     def test_set_domain(self):
#         with self.test_session():
#             with self.assertRaises(AssertionError):
#                 super(TestBayesianOptimizer, self).test_set_domain()
#
#             domain = gpflowopt.domain.ContinuousParameter("x1", -2.0, 2.0) + \
#                      gpflowopt.domain.ContinuousParameter("x2", -2.0, 2.0)
#             self.optimizer.domain = domain
#             expected = gpflowopt.design.LatinHyperCube(16, self.domain).generate() / 4 + 0.5
#             self.assertTrue(np.allclose(expected, self.optimizer.acquisition.models[0].wrapped.X.value))
#
#
# class TestBayesianOptimizerConfigurations(GPflowOptTestCase):
#     def setUp(self):
#         self.domain = gpflowopt.domain.ContinuousParameter("x1", 0.0, 1.0) + \
#                       gpflowopt.domain.ContinuousParameter("x2", 0.0, 1.0)
#         self.acquisition = gpflowopt.acquisition.ExpectedImprovement(create_parabola_model(self.domain))
#
#     def test_initial_design(self):
#         with self.test_session():
#             design = gpflowopt.design.RandomDesign(5, self.domain)
#             optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, initial=design)
#
#             result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=0)
#             self.assertTrue(result.success)
#             self.assertEqual(result.nfev, 5, "Evaluated only initial")
#             self.assertTupleEqual(optimizer.acquisition.data[0].shape, (21, 2))
#             self.assertTupleEqual(optimizer.acquisition.data[1].shape, (21, 1))
#
#             result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=0)
#             self.assertTrue(result.success)
#             self.assertEqual(result.nfev, 0, "Initial was not reset")
#             self.assertTupleEqual(optimizer.acquisition.data[0].shape, (21, 2))
#             self.assertTupleEqual(optimizer.acquisition.data[1].shape, (21, 1))
#
#     def test_mcmc(self):
#         with self.test_session():
#             optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, hyper_draws=10)
#             self.assertIsInstance(optimizer.acquisition, gpflowopt.acquisition.MCMCAcquistion)
#             self.assertEqual(len(optimizer.acquisition.operands), 10)
#             self.assertEqual(optimizer.acquisition.operands[0], self.acquisition)
#
#             result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=20)
#             self.assertTrue(result.success)
#             self.assertTrue(np.allclose(result.x, 0), msg="Optimizer failed to find optimum")
#             self.assertTrue(np.allclose(result.fun, 0), msg="Incorrect function value returned")
#
#     def test_callback(self):
#         class DummyCallback(object):
#             def __init__(self):
#                 self.counter = 0
#
#             def __call__(self, models):
#                 self.counter += 1
#
#         c = DummyCallback()
#         optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, callback=c)
#         result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=2)
#         self.assertEqual(c.counter, 2)
#
#     def test_callback_recompile(self):
#         class DummyCallback(object):
#             def __init__(self):
#                 self.recompile = False
#
#             def __call__(self, models):
#                 c = np.random.randint(2, 10)
#                 models[0].kern.variance.prior = gpflow.priors.Gamma(c, 1./c)
#                 self.recompile = models[0]._needs_recompile
#
#         c = DummyCallback()
#         optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, callback=c)
#         self.acquisition.evaluate(np.zeros((1,2))) # Make sure its run and setup to skip
#         result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
#         self.assertFalse(c.recompile)
#         result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
#         self.assertTrue(c.recompile)
#         self.assertFalse(self.acquisition.models[0]._needs_recompile)
#
#     def test_callback_recompile_mcmc(self):
#         class DummyCallback(object):
#             def __init__(self):
#                 self.no_models = 0
#
#             def __call__(self, models):
#                 c = np.random.randint(2, 10)
#                 models[0].kern.variance.prior = gpflow.priors.Gamma(c, 1. / c)
#                 self.no_models = len(models)
#
#         c = DummyCallback()
#         optimizer = gpflowopt.BayesianOptimizer(self.domain, self.acquisition, hyper_draws=5, callback=c)
#         opers = optimizer.acquisition.operands
#         result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
#         self.assertEqual(c.no_models, 1)
#         self.assertEqual(id(opers[0]), id(optimizer.acquisition.operands[0]))
#         for op1, op2 in zip(opers[1:], optimizer.acquisition.operands[1:]):
#             self.assertNotEqual(id(op1), id(op2))
#         opers = optimizer.acquisition.operands
#         result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
#         self.assertEqual(id(opers[0]), id(optimizer.acquisition.operands[0]))
#         for op1, op2 in zip(opers[1:], optimizer.acquisition.operands[1:]):
#             self.assertNotEqual(id(op1), id(op2))
#
#     def test_nongpr_model(self):
#         design = gpflowopt.design.LatinHyperCube(16, self.domain)
#         X, Y = design.generate(), parabola2d(design.generate())[0]
#         m = gpflow.vgp.VGP(X, Y, gpflow.kernels.RBF(2, ARD=True), likelihood=gpflow.likelihoods.Gaussian())
#         acq = gpflowopt.acquisition.ExpectedImprovement(m)
#         optimizer = gpflowopt.BayesianOptimizer(self.domain, acq)
#         result = optimizer.optimize(lambda X: parabola2d(X)[0], n_iter=1)
#         self.assertTrue(result.success)
#
#
# class TestSilentOptimization(GPflowOptTestCase):
#     @contextmanager
#     def captured_output(self):
#         # Captures all stdout/stderr
#         new_out, new_err = six.StringIO(), six.StringIO()
#         old_out, old_err = sys.stdout, sys.stderr
#         try:
#             sys.stdout, sys.stderr = new_out, new_err
#             yield sys.stdout, sys.stderr
#         finally:
#             sys.stdout, sys.stderr = old_out, old_err
#
#     def test_silent(self):
#         class EmittingOptimizer(gpflowopt.optim.Optimizer):
#             def __init__(self):
#                 super(EmittingOptimizer, self).__init__(gpflowopt.domain.ContinuousParameter('x0', 0, 1))
#
#             def _optimize(self, objective):
#                 print('hello world!')
#                 return OptimizeResult(x=np.array([0.5]))
#
#         # First, optimize with silent mode off. Should return the stdout of the optimizer
#         opt = EmittingOptimizer()
#         with self.captured_output() as (out, err):
#             opt.optimize(None)
#             output = out.getvalue().strip()
#             self.assertEqual(output, 'hello world!')
#
#         # Now with silent mode on
#         with self.captured_output() as (out, err):
#             with opt.silent():
#                 opt.optimize(None)
#                 output = out.getvalue().strip()
#                 self.assertEqual(output, '')
#
