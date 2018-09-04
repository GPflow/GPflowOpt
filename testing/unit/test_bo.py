import os
import gpflow
import gpflowopt
import numpy as np
import pytest
from ..utility import parabola2d, KeyboardRaiser, vlmop2


class TestBayesianOptimizer(object):

    @pytest.fixture()
    def acquisition(self, parabola_model):
        yield gpflowopt.acquisition.ExpectedImprovement(parabola_model)

    @pytest.fixture(params=(False, True))
    def optimizer(self, request, domain, acquisition):
        yield gpflowopt.BayesianOptimizer(domain, acquisition, verbose=request.param)

    def test_default_initial(self, optimizer):
        assert optimizer._initial.shape == (0, 2)

    def test_set_domain(self, domain, optimizer):
        with pytest.raises(AssertionError):
            optimizer.domain = gpflowopt.domain.UnitCube(3)
        target_domain = gpflowopt.domain.ContinuousParameter("x1", -2.0, 2.0) + \
                        gpflowopt.domain.ContinuousParameter("x2", -2.0, 2.0)
        optimizer.domain = target_domain
        expected = gpflowopt.design.LatinHyperCube(16, domain).generate() / 4 + 0.5
        np.testing.assert_allclose(expected, optimizer.acquisition.models[0].wrapped.X.read_value())

    def test_optimize(self, optimizer):
        optimizer.acquisition.optimize_restarts = 2
        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=5)
        assert result.success
        assert result.nfev == 5
        np.testing.assert_allclose(result.x, 0)
        np.testing.assert_allclose(result.fun, 0)

    def test_optimizer_interrupt(self, optimizer):
        result = optimizer.optimize(KeyboardRaiser(3, lambda X: parabola2d(X)), n_iter=20)
        assert not result.success
        np.testing.assert_allclose(result.x, 0.0)

    def test_optimize_multi_objective(self, domain, vlmop2_models):
        m1, m2 = vlmop2_models
        acquisition = gpflowopt.acquisition.ExpectedImprovement(m1) + gpflowopt.acquisition.ExpectedImprovement(m2)
        optimizer = gpflowopt.BayesianOptimizer(domain, acquisition)
        result = optimizer.optimize(vlmop2, n_iter=2)
        assert result.success
        assert result.nfev == 2
        assert result.x.shape == (7, 2)
        assert result.fun.shape == (7, 2)
        _, dom = gpflowopt.pareto.non_dominated_sort(result.fun)
        assert np.all(dom == 0)

    def test_optimize_constraint(self, domain, parabola_model):
        acquisition = gpflowopt.acquisition.ProbabilityOfFeasibility(parabola_model, threshold=-1.0)
        optimizer = gpflowopt.BayesianOptimizer(domain, acquisition, verbose=True)
        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=1)
        assert not result.success
        assert result.message == 'No evaluations satisfied all the constraints'
        assert result.nfev == 1
        assert result.x.shape == (17, 2)
        assert result.fun.shape == (17, 0)
        assert result.constraints.shape == (17, 1)

        acquisition.threshold.assign(0.4)
        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=1)
        assert result.success
        assert result.nfev == 1
        assert result.x.shape == (6, 2)
        assert result.fun.shape == (6, 0)
        assert result.constraints.shape == (6, 1)

    def test_failsafe(self, optimizer):
        X, Y = optimizer.acquisition.data
        # Provoke cholesky faillure
        optimizer.acquisition.optimize_restarts = 1
        optimizer.acquisition.clear()
        optimizer.acquisition.models[0].likelihood.variance.transform = gpflow.transforms.Identity()
        optimizer.acquisition.models[0].likelihood.variance = -5.0
        optimizer.acquisition.compile()
        with pytest.raises(RuntimeError) as e:
            with optimizer.failsafe():
                optimizer.acquisition._optimize_models()

        fname = 'failed_bopt_{0}.npz'.format(id(e.value))
        assert os.path.isfile(fname)
        with np.load(fname) as data:
            np.testing.assert_almost_equal(data['X'], X)
            np.testing.assert_almost_equal(data['Y'], Y)
        os.remove(fname)

    def test_initial_design(self, domain, acquisition):
        design = gpflowopt.design.RandomDesign(5, domain)
        optimizer = gpflowopt.BayesianOptimizer(domain, acquisition, initial=design)

        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=0)
        assert result.success
        assert result.nfev == 5
        assert optimizer.acquisition.data[0].shape == (21, 2)
        assert optimizer.acquisition.data[1].shape == (21, 1)

        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=0)
        assert result.success
        assert result.nfev == 0
        assert optimizer.acquisition.data[0].shape == (21, 2)
        assert optimizer.acquisition.data[1].shape == (21, 1)

    def test_callback(self, domain, acquisition):
        class DummyCallback(object):
            def __init__(self):
                self.counter = 0

            def __call__(self, trainables):
                self.counter += 1

        c = DummyCallback()
        optimizer = gpflowopt.BayesianOptimizer(domain, acquisition, callback=c)
        _ = optimizer.optimize(lambda X: parabola2d(X), n_iter=2)
        assert c.counter == 2

    def test_mcmc(self, domain, acquisition):
        optimizer = gpflowopt.BayesianOptimizer(domain, acquisition, hyper_draws=2)
        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=2)
        assert result.success
        np.testing.assert_allclose(result.x, 0)
        np.testing.assert_allclose(result.fun, 0)

    def test_nongpr_model(self, domain):
        design = gpflowopt.design.LatinHyperCube(16, domain)
        X, Y = design.generate(), parabola2d(design.generate())
        m = gpflow.models.VGP(X, Y, gpflow.kernels.RBF(2, ARD=True), likelihood=gpflow.likelihoods.Gaussian())
        acq = gpflowopt.acquisition.ExpectedImprovement(m)
        optimizer = gpflowopt.BayesianOptimizer(domain, acq)
        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=1)
        assert result.success
