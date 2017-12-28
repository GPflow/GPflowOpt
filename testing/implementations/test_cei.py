import numpy as np
import gpflowopt
import gpflow
from ..utility import parabola2d


def test_constrained_ei(domain, session):
    design = gpflowopt.design.LatinHyperCube(16, domain)
    X = design.generate()
    Yo = parabola2d(X)
    Yc = -parabola2d(X) + 0.5
    m1 = gpflow.models.GPR(X, Yo, gpflow.kernels.RBF(2, ARD=False, lengthscales=37.7554549981, variance=845886.3367827121))
    m1.likelihood.variance = 1e-6
    m2 = gpflow.models.GPR(X, Yc, gpflow.kernels.RBF(2, ARD=False, lengthscales=0.851406328779, variance=845886.3367827121))
    m2.likelihood.variance = 1e-6
    ei = gpflowopt.acquisition.ExpectedImprovement(m1)
    pof = gpflowopt.acquisition.ProbabilityOfFeasibility(m2)
    joint = ei * pof

    # Test output indices
    np.testing.assert_allclose(joint.objective_indices(), np.array([0], dtype=int))
    np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))

    # Test proper setup
    joint._needs_setup = False
    joint._setup()
    assert ei.fmin.read_value() > np.min(ei.data[1])
    np.testing.assert_allclose(ei.fmin.read_value(), np.min(ei.data[1][pof.feasible_data_index(), :]), atol=1e-3)
