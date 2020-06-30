import numpy as np
import gpflow
import gpflowopt
import os
import tensorflow as tf


class GPflowOptTestCase(tf.test.TestCase):
    """
    Wrapper for TestCase to avoid massive duplication of resetting
    Tensorflow Graph.
    """

    _multiprocess_can_split_ = True

    def tearDown(self):
        tf.reset_default_graph()
        super(GPflowOptTestCase, self).tearDown()


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T


def plane(X):
    return X[:, [0]] - 0.5


def vlmop2(x):
    transl = 1 / np.sqrt(2)
    part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
    part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
    y1 = 1 - np.exp(-1 * part1)
    y2 = 1 - np.exp(-1 * part2)
    return np.hstack((y1, y2))


def load_data(file):
    path = os.path.dirname(os.path.realpath(__file__))
    return np.load(os.path.join(path, 'data', file))


def create_parabola_model(domain, design=None):
    if design is None:
        design = gpflowopt.design.LatinHyperCube(16, domain)
    X, Y = design.generate(), parabola2d(design.generate())
    m = gpflow.gpr.GPR(X, Y, gpflow.kernels.RBF(2, ARD=True))
    return m


def create_plane_model(domain, design=None):
    if design is None:
        design = gpflowopt.design.LatinHyperCube(25, domain)
    X, Y = design.generate(), plane(design.generate())
    m = gpflow.gpr.GPR(X, Y, gpflow.kernels.RBF(2, ARD=True))
    return m


def create_vlmop2_model():
    data = load_data('vlmop.npz')
    m1 = gpflow.gpr.GPR(data['X'], data['Y'][:, [0]], kern=gpflow.kernels.Matern32(2))
    m2 = gpflow.gpr.GPR(data['X'], data['Y'][:, [1]], kern=gpflow.kernels.Matern32(2))
    return [m1, m2]