import sys
import io
import gpflowopt
import numpy as np
import warnings
import pytest
from contextlib import contextmanager
from scipy.optimize import OptimizeResult
from ..utility import parabola2d_grad, KeyboardRaiser


@pytest.fixture(scope="module")
def design(domain):
    yield gpflowopt.design.FactorialDesign(4, domain)


class TestCandidateOptimizer(object):

    @pytest.fixture()
    @pytest.mark.usefixtures("session")
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
    @pytest.mark.usefixtures("session")
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
            np.testing.assert_allclose(result.x, 0, atol=1e-3)
            np.testing.assert_allclose(result.fun, 0, atol=1e-3)

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


class TestSilentOptimization(object):
    @contextmanager
    def captured_output(self):
        # Captures all stdout/stderr
        new_out, new_err = io.StringIO(), io.StringIO()
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
            assert output == 'hello world!'

        # Now with silent mode on
        with self.captured_output() as (out, err):
            with opt.silent():
                opt.optimize(None)
                output = out.getvalue().strip()
                assert output == ''


#