import os
import gpflow
import gpflowopt
import numpy as np
import pytest
from ..utility import parabola2d, KeyboardRaiser, vlmop2


class TestBayesianOptimizer(object):

    @pytest.fixture()
    def optimizer(self, domain, parabola_model):
        acquisition = gpflowopt.acquisition.ExpectedImprovement(parabola_model)
        yield gpflowopt.BayesianOptimizer(domain, acquisition)

    def test_default_initial(self, optimizer):
        assert optimizer._initial.shape == (0, 2)

    def test_set_domain(self, domain, optimizer):
        with pytest.raises(AssertionError):
            optimizer.domain = gpflowopt.domain.UnitCube(3)
        domain = gpflowopt.domain.ContinuousParameter("x1", -2.0, 2.0) + \
                 gpflowopt.domain.ContinuousParameter("x2", -2.0, 2.0)
        optimizer.domain = domain
        expected = gpflowopt.design.LatinHyperCube(16, domain).generate() / 4 + 0.5
        np.testing.assert_allclose(expected, optimizer.acquisition.models[0].wrapped.X.read_value())

    def test_optimize(self, optimizer):
        result = optimizer.optimize(lambda X: parabola2d(X), n_iter=20)
        assert result.success
        assert result.nfev == 20
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

        fname = 'failed_bopt_{0}.npz'.format(id(e.exception))
        assert os.path.isfile(fname)
        with np.load(fname) as data:
            np.testing.assert_almost_equal(data['X'], X)
            np.testing.assert_almost_equal(data['Y'], Y)
        os.remove(fname)

