import gpflowopt
import numpy as np
from parameterized import parameterized
from .utility import GPflowOptTestCase


# This is what we expect the versions applying the decorators to produce (simple additions)
def ref_function(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True), X

# For to_kwargs
domain = gpflowopt.domain.ContinuousParameter('x', 0, 1) + gpflowopt.domain.ContinuousParameter('y', 0, 1)


# Some versions
@gpflowopt.objective.to_args
def add_to_args(x, y):
    return ref_function(np.vstack((x, y)).T)


@gpflowopt.objective.to_kwargs(domain)
def add_to_kwargs(x=None, y=None):
    return ref_function(np.vstack((x, y)).T)


@gpflowopt.objective.batch_apply
def add_batch_apply(Xflat):
    f, g = ref_function(Xflat)
    return f, g[0, :]


@gpflowopt.objective.batch_apply
def add_batch_apply_no_dims(Xflat):
    return np.sum(Xflat), Xflat


@gpflowopt.objective.batch_apply
@gpflowopt.objective.to_args
def add_batch_apply_to_args(x, y):
    f, g = ref_function(np.vstack((x, y)).T)
    return f, g[0, :]


@gpflowopt.objective.batch_apply
@gpflowopt.objective.to_kwargs(domain)
def add_batch_apply_to_kwargs(x=None, y=None):
    f, g = ref_function(np.vstack((x, y)).T)
    return f, g[0, :]


@gpflowopt.objective.batch_apply
def triple_objective(Xflat):
    f1, g1 = ref_function(Xflat)
    f2, g2 = ref_function(2 * Xflat)
    f3, g3 = ref_function(0.5 * Xflat)
    return np.hstack((f1, f2, f3)), np.vstack((g1, g2, g3)).T


@gpflowopt.objective.batch_apply
def add_batch_apply_no_grad(Xflat):
    f, g = ref_function(Xflat)
    return f


class TestDecorators(GPflowOptTestCase):

    @staticmethod
    def check_reference(f, g, X):
        np.testing.assert_almost_equal(f, ref_function(X)[0])
        np.testing.assert_almost_equal(g, ref_function(X)[1])

    @parameterized.expand([(add_to_args,),
                           (add_to_kwargs,),
                           (add_batch_apply,),
                           (add_batch_apply_no_dims,),
                           (add_batch_apply_to_args,),
                           (add_batch_apply_to_kwargs,)])
    def test_one_point(self, fun):
        X = np.random.rand(2)
        f, g = fun(X)
        self.assertTupleEqual(f.shape, (1, 1))
        self.assertTupleEqual(g.shape, (1, 2))
        self.__class__.check_reference(f, g, X)

    @parameterized.expand([(add_to_args,),
                           (add_to_kwargs,),
                           (add_batch_apply,),
                           (add_batch_apply_no_dims,),
                           (add_batch_apply_to_args,),
                           (add_batch_apply_to_kwargs,)])
    def test_multiple_points(self, fun):
        X = np.random.rand(5, 2)
        f, g = fun(X)
        self.assertTupleEqual(f.shape, (5, 1))
        self.assertTupleEqual(g.shape, (5, 2))
        self.__class__.check_reference(f, g, X)

    def test_multiple_objectives(self):
        X = np.random.rand(5, 2)
        f, g = triple_objective(X)
        self.assertTupleEqual(f.shape, (5, 3))
        self.assertTupleEqual(g.shape, (5, 2, 3))
        self.__class__.check_reference(f[:, [0]], g[..., 0], X)
        self.__class__.check_reference(f[:, [1]], g[..., 1], 2*X)
        self.__class__.check_reference(f[:, [2]], g[..., 2], 0.5*X)

    def test_no_grad(self):
        X = np.random.rand(5, 2)
        f = add_batch_apply_no_grad(X)
        self.assertTupleEqual(f.shape, (5, 1))
        np.testing.assert_almost_equal(f, ref_function(X)[0])

