import gpflowopt
import numpy as np
from ..utility import GPflowOptTestCase


class TestContinuousParameter(GPflowOptTestCase):

    def test_simple(self):
        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1)
        self.assertTrue(np.allclose(p._range, [0,1]), msg="Internal storage of object incorrect")
        self.assertEqual(p.lower, 0, msg="Lower should equal 0")
        self.assertEqual(p.upper, 1, msg="Upper should equal 1")
        self.assertEqual(p.size, 1, msg="Size of parameter should equal 1")

        p.upper = 2
        self.assertEqual(p.upper, 2, msg="After assignment, upper should equal 2")
        p.lower = 1
        self.assertEqual(p.lower, 1, msg="After assignment, lower should equal 2")

        p = np.sum([gpflowopt.domain.ContinuousParameter("x1", 0, 1)])
        self.assertTrue(p.size == 1, msg="Construction of domain by list using sum failed")

    def test_equality(self):
        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1)
        pne = gpflowopt.domain.ContinuousParameter("x1", 0, 2)
        self.assertNotEqual(p, pne, msg="Should not be equal (invalid upper)")
        pne = gpflowopt.domain.ContinuousParameter("x1", -1, 1)
        self.assertNotEqual(p, pne, msg="Should not be equal (invalid lower)")
        pne = gpflowopt.domain.ContinuousParameter("x1", -1, 2)
        self.assertNotEqual(p, pne, msg="Should not be equal (invalid lower/upper)")
        p.lower = -1
        p.upper = 2
        self.assertEqual(p, pne, msg="Should be equal after adjusting bounds")

    def test_indexing(self):
        p = np.sum([gpflowopt.domain.ContinuousParameter("x1", 0, 1),
                    gpflowopt.domain.ContinuousParameter("x2", 0, 1),
                    gpflowopt.domain.ContinuousParameter("x3", 0, 1),
                    gpflowopt.domain.ContinuousParameter("x4", 0, 1)])

        subdomain = p[['x4', 'x1', 2]]
        self.assertTrue(subdomain.size == 3, msg="Subdomain should have size 3")
        self.assertTrue(subdomain[0].label == 'x4', msg="Subdomain's first parameter should be 'x4'")
        self.assertTrue(subdomain[1].label == 'x1', msg="Subdomain's second parameter should be 'x1'")
        self.assertTrue(subdomain[2].label == 'x3', msg="Subdomain's third parameter should be 'x3'")

    def test_containment(self):
        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1)
        self.assertIn(0, p, msg="Point is within domain")
        self.assertIn(0.5, p, msg="Point is within domain")
        self.assertIn(1, p, msg="Point is within domain")
        self.assertNotIn(1.1, p, msg="Point is not within domain")
        self.assertNotIn(-0.5, p, msg="Point is not within domain")

    def test_value(self):
        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1)
        self.assertTupleEqual(p.value.shape, (1,), msg="Default value has incorrect shape.")
        self.assertTrue(np.allclose(p.value, 0.5), msg="Parameter has incorrect default value")

        p.value = 0.8
        self.assertTrue(np.allclose(p.value, 0.8), msg="Parameter has incorrect value after update")

        p.value = [0.6, 0.8]
        self.assertTupleEqual(p.value.shape, (2,), msg="Default value has incorrect shape.")
        np.testing.assert_allclose(p.value, np.array([0.6, 0.8]), err_msg="Parameter has incorrect value after update")

        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1, 0.2)
        self.assertTupleEqual(p.value.shape, (1,), msg="Default value has incorrect shape.")
        self.assertTrue(np.allclose(p.value, 0.2), msg="Parameter has incorrect initialized value")


class TestHypercubeDomain(GPflowOptTestCase):

    def setUp(self):
        self.domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 4)])

    def test_object_integrity(self):
        self.assertEqual(len(self.domain._parameters), 3)

    def test_simple(self):
        self.assertEqual(self.domain.size, 3, msg="Size of domain should equal 3")
        self.assertTrue(np.allclose(self.domain.lower, -1.0), msg="Lower of domain should equal -1 for all parameters")
        self.assertTrue(np.allclose(self.domain.upper, 1.0), msg="Lower of domain should equal 1 for all parameters")

    def test_equality(self):
        dne = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                     [gpflowopt.domain.ContinuousParameter("x3", -3, 1)])
        self.assertNotEqual(self.domain, dne, msg="One lower bound mismatch, should not be equal.")
        dne = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                     [gpflowopt.domain.ContinuousParameter("x3", -1, 2)])
        self.assertNotEqual(self.domain, dne, msg="One upper bound mismatch, should not be equal.")
        dne = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.assertNotEqual(self.domain, dne, msg="Size mismatch")
        de = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 4)])
        self.assertEqual(self.domain, de, msg="No mismatches, should be equal")

    def test_parenting(self):
        for p in self.domain:
            self.assertEqual(id(p._parent), id(self.domain), "Misspecified parent link detected")

    def test_access(self):
        for i in range(self.domain.size):
            self.assertEqual(self.domain[i].label, "x{0}".format(i+1), "Accessing parameters, encountering "
                                                                     "incorrect labels")

        self.domain[2].lower = -2
        de = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                    [gpflowopt.domain.ContinuousParameter("x3", -2, 1)])

        self.assertEqual(self.domain, de, msg="No mismatches, should be equal")

    def test_containment(self):
        A = np.random.rand(50,3)*2-1
        self.assertTrue(A in self.domain, msg="Generated random points within domain")

        A = np.vstack((A, np.array([-2, -2, -2])))
        self.assertFalse(A in self.domain, msg="One of the points was not in the domain")

        A = np.random.rand(50,4)*2-1
        self.assertFalse(A in self.domain, msg="Generated random points have different dimensionality")

    def test_value(self):
        self.assertTupleEqual(self.domain.value.shape, (1, 3), msg="Default value has incorrect shape.")
        np.testing.assert_allclose(self.domain.value, np.array([[0, 0, 0]]), err_msg="Parameter has incorrect initial value")

        A = np.random.rand(10, 3) * 2 - 1
        self.domain.value = A
        self.assertTupleEqual(self.domain.value.shape, (10, 3), msg="Assigned value has incorrect shape.")
        np.testing.assert_allclose(self.domain.value, A, err_msg="Parameter has incorrect value after assignment")

    def test_transformation(self):
        X = np.random.rand(50,3)*2-1
        target = gpflowopt.domain.UnitCube(3)
        transform = self.domain >> target
        self.assertTrue(np.allclose(transform.forward(X), (X + 1) / 2), msg="Transformation to [0,1] incorrect")
        self.assertTrue(np.allclose(transform.backward(transform.forward(X)), X),
                        msg="Transforming back and forth yields different result")

        inv_transform = target >> self.domain
        self.assertTrue(np.allclose(transform.backward(transform.forward(X)),
                                    inv_transform.forward(transform.forward(X))),
                        msg="Inverse transform yields different results")
        self.assertTrue(np.allclose((~transform).A.value, inv_transform.A.value))
        self.assertTrue(np.allclose((~transform).b.value, inv_transform.b.value))

    def test_unitcube(self):
        domain = gpflowopt.domain.UnitCube(3)
        self.assertTrue(np.allclose(domain.lower, 0))
        self.assertTrue(np.allclose(domain.upper, 1))
        self.assertEqual(domain.size, 3)

