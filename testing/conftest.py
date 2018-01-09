import os
import numpy as np
import gpflow
import gpflowopt
import pytest
import tensorflow as tf
from .utility import parabola2d, plane


@pytest.fixture(scope="session")
def domain():
    return np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])


@pytest.fixture()
def session():
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            yield sess


@pytest.fixture()
@pytest.mark.usefixtures("session")
def parabola_model(domain):
    design = gpflowopt.design.LatinHyperCube(16, domain)
    X, Y = design.generate(), parabola2d(design.generate())
    m = gpflow.models.GPR(X, Y, gpflow.kernels.RBF(2, ARD=True))
    yield m


@pytest.fixture()
@pytest.mark.usefixtures("session")
def plane_model(domain):
    design = gpflowopt.design.LatinHyperCube(25, domain)
    X, Y = design.generate(), plane(design.generate())
    m = gpflow.models.GPR(X, Y, gpflow.kernels.RBF(2, ARD=True))
    yield m


@pytest.fixture(scope="module")
def vlmop2_data():
    path = os.path.dirname(os.path.realpath(__file__))
    return np.load(os.path.join(path, 'data', 'vlmop.npz'))


@pytest.fixture()
@pytest.mark.usefixtures("session")
def vlmop2_models(vlmop2_data):
    m1 = gpflow.models.GPR(vlmop2_data['X'], vlmop2_data['Y'][:, [0]], kern=gpflow.kernels.Matern32(2))
    m2 = gpflow.models.GPR(vlmop2_data['X'], vlmop2_data['Y'][:, [1]], kern=gpflow.kernels.Matern32(2))
    yield [m1, m2]
