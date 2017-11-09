import gpflow
import gpflowopt
import numpy as np
from ..utility import GPflowOptTestCase


class TestRecompile(GPflowOptTestCase):
    """
    Regression test for #37
    """
    def test_vgp(self):
        with self.test_session():
            domain = gpflowopt.domain.UnitCube(2)
            X = gpflowopt.design.RandomDesign(10, domain).generate()
            Y = np.sin(X[:,[0]])
            m = gpflow.vgp.VGP(X, Y, gpflow.kernels.RBF(2), gpflow.likelihoods.Gaussian())
            acq = gpflowopt.acquisition.ExpectedImprovement(m)
            m.compile()
            self.assertFalse(m._needs_recompile)
            acq.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())
            self.assertTrue(hasattr(acq, '_evaluate_AF_storage'))

            Xnew = gpflowopt.design.RandomDesign(5, domain).generate()
            Ynew = np.sin(Xnew[:,[0]])
            acq.set_data(np.vstack((X, Xnew)), np.vstack((Y, Ynew)))
            self.assertFalse(hasattr(acq, '_needs_recompile'))
            self.assertFalse(hasattr(acq, '_evaluate_AF_storage'))
            acq.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())