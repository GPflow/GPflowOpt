import numpy as np
import gpflow
import gpflowopt


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


def parabola2d(X):
    return np.atleast_2d(np.sum(X ** 2, axis=1)).T


def parabola2d_grad(X):
    return parabola2d(X), 2 * X


def plane(X):
    return X[:, [0]] - 0.5


def vlmop2(x):
    transl = 1 / np.sqrt(2)
    part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
    part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
    y1 = 1 - np.exp(-1 * part1)
    y2 = 1 - np.exp(-1 * part2)
    return np.hstack((y1, y2))


def create_parabola_model(domain, design=None):
    if design is None:
        design = gpflowopt.design.LatinHyperCube(16, domain)
    X, Y = design.generate(), parabola2d(design.generate())
    m = gpflow.models.GPR(X, Y, gpflow.kernels.RBF(2, ARD=True))
    return m

