import gpflowopt
import numpy as np
import os
import tensorflow as tf
import pytest
from functools import partial

designs_to_test = [(gpflowopt.design.RandomDesign, (200,)),
                   (gpflowopt.design.EmptyDesign, tuple()),
                   (gpflowopt.design.FactorialDesign, (4,)),
                   (gpflowopt.design.LatinHyperCube, (20,))]


@pytest.fixture(scope='module',
                params=range(1, 6))
def domain(request):
    j = request.param
    return np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -i, 2 * i) for i in range(1, j + 1)])


@pytest.fixture(scope='module',
                params=designs_to_test)
def design_part(request):
    cls, args = request.param
    return partial(cls, *args)


def test_design_compliance(domain, design_part):
    with tf.Session(graph=tf.Graph()):
        design = design_part(domain)
        points = design.generate()
        assert points.shape == (design.size, domain.size)
        assert points in domain


def test_create_to_generate(domain, design_part):
    if design_part.func == gpflowopt.design.RandomDesign:
        pytest.skip("create vs generate does not work for RandomDesign")
        return

    with tf.Session(graph=tf.Graph()):
        design = design_part(domain)
        X = design.create_design()
        Xt = design.generate()
        transform = design.generative_domain >> design.domain
        Xs = transform.forward(X)
        Xr = transform.backward(Xt)

        np.testing.assert_allclose(Xs, Xt, atol=1e-4, err_msg="Incorrect scaling from generative domain to domain")
        np.testing.assert_allclose(Xr, X, atol=1e-4, err_msg="Incorrect scaling from generative domain to domain")


def test_factorial_validity(domain):
    design = gpflowopt.design.FactorialDesign(4, domain)
    with tf.Session(graph=tf.Graph()):
        A = design.generate()
        for i, l, u in zip(range(1, design.domain.size + 1), design.domain.lower, design.domain.upper):
            assert np.all(np.any(np.abs(A[:,i - 1] - np.linspace(l, u, 4)[:, None]) < 1e-4, axis=0))


def test_lhd_validity(domain):
    groundtruths = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'lhd.npz'))
    lhd = gpflowopt.design.LatinHyperCube(20, domain)
    with tf.Session(graph=tf.Graph()):
        points = lhd.generate()
        lhds = list(map(lambda file: groundtruths[file], groundtruths.files))
        idx = np.argsort([lhd.shape[-1] for lhd in lhds])
        np.testing.assert_allclose(points, lhds[idx[domain.size-1]])
