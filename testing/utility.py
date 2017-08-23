import numpy as np
import GPflow
import GPflowOpt
import os


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
        design = GPflowOpt.design.LatinHyperCube(16, domain)
    X, Y = design.generate(), parabola2d(design.generate())
    m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
    return m


def create_plane_model(domain, design=None):
    if design is None:
        design = GPflowOpt.design.LatinHyperCube(25, domain)
    X, Y = design.generate(), plane(design.generate())
    m = GPflow.gpr.GPR(X, Y, GPflow.kernels.RBF(2, ARD=True))
    return m


def create_vlmop2_model():
    data = load_data('vlmop.npz')
    m1 = GPflow.gpr.GPR(data['X'], data['Y'][:, [0]], kern=GPflow.kernels.Matern32(2))
    m2 = GPflow.gpr.GPR(data['X'], data['Y'][:, [1]], kern=GPflow.kernels.Matern32(2))
    return [m1, m2]