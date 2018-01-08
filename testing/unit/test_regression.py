import gpflow
import gpflowopt
import numpy as np


def test_vgp(session):
    """
    Regression test for #37
    """
    domain = gpflowopt.domain.UnitCube(2)
    X = gpflowopt.design.RandomDesign(10, domain).generate()
    Y = np.sin(X[:,[0]])
    m = gpflow.models.VGP(X, Y, gpflow.kernels.RBF(2), gpflow.likelihoods.Gaussian())
    acq = gpflowopt.acquisition.ExpectedImprovement(m)
    acq.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())

    Xnew = gpflowopt.design.RandomDesign(5, domain).generate()
    Ynew = np.sin(Xnew[:,[0]])
    acq.set_data(np.vstack((X, Xnew)), np.vstack((Y, Ynew)))

    gpflowopt.bo.default_callback(m, gpflow.train.ScipyOptimizer())
    acq.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())