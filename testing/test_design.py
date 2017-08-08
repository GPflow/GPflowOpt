import GPflowOpt
import unittest
import numpy as np
import os


class _TestDesign(object):

    _multiprocess_can_split_ = True

    @property
    def designs(self):
        raise NotImplementedError()

    @property
    def domains(self):
        createfx = lambda j: np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -i, 2 * i) for i in range(1, j+1)])
        return list(map(createfx, np.arange(1, 6)))

    def test_design_compliance(self):
        points = [design.generate() for design in self.designs]
        for p, d in zip(points, self.designs):
            self.assertTupleEqual(p.shape, (d.size, d.domain.size), msg="Generated design does match specifications")
            self.assertIn(p, d.domain, "Not all generated points are generated within the domain")

    def test_create_to_generate(self):
        X = [design.create_design() for design in self.designs]
        Xt = [design.generate() for design in self.designs]
        transforms = [design.generative_domain >> design.domain for design in self.designs]

        Xs = [t.forward(p) for p, t in zip(X, transforms)]
        Xr = [t.backward(p) for p, t in zip(Xt, transforms)]

        for generated, scaled in zip(Xs, Xt):
            np.testing.assert_allclose(generated, scaled, atol=1e-4,
                                       err_msg="Incorrect scaling from generative domain to domain")

        for generated, scaled in zip(Xr, X):
            np.testing.assert_allclose(generated, scaled, atol=1e-4,
                                       err_msg="Incorrect scaling from generative domain to domain")


class TestRandomDesign(_TestDesign, unittest.TestCase):

    @_TestDesign.designs.getter
    def designs(self):
        return [GPflowOpt.design.RandomDesign(200, domain) for domain in self.domains]

    def test_create_to_generate(self):
        pass


class TestEmptyDesign(_TestDesign, unittest.TestCase):
    @_TestDesign.designs.getter
    def designs(self):
        return [GPflowOpt.design.EmptyDesign(domain) for domain in self.domains]


class TestFactorialDesign(_TestDesign, unittest.TestCase):
    @_TestDesign.designs.getter
    def designs(self):
        return [GPflowOpt.design.FactorialDesign(4, domain) for domain in self.domains]

    def test_validity(self):
        for design in self.designs:
            A = design.generate()
            for i, l, u in zip(range(1, design.domain.size + 1), design.domain.lower, design.domain.upper):
                self.assertTrue(np.all(np.any(np.abs(A[:,i - 1] - np.linspace(l, u, 4)[:, None]) < 1e-4, axis=0)),
                                msg="Generated off-grid.")


class TestLatinHyperCubeDesign(_TestDesign, unittest.TestCase):

    @_TestDesign.designs.getter
    def designs(self):
        return [GPflowOpt.design.LatinHyperCube(20, domain) for domain in self.domains]

    def test_validity(self):
        groundtruth = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'lhd.npz'))
        points = [lhd.generate() for lhd in self.designs]
        lhds = map(lambda file: groundtruth[file], groundtruth.files)
        idx = np.argsort([lhd.shape[-1] for lhd in lhds])
        for generated, real in zip(points, map(lambda file: groundtruth[file], np.array(groundtruth.files)[idx])):
            self.assertTrue(np.allclose(generated, real), msg="Generated LHD does not correspond to the groundtruth")