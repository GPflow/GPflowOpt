import GPflowOpt
import unittest
import numpy as np


class TestContinuousParameter(unittest.TestCase):

    def test_simple(self):
        p = GPflowOpt.domain.ContinuousParameter("x1", 0, 1)
        self.assertListEqual(p._range, [0,1], msg="Internal storage of object incorrect")
        self.assertEqual(p.lower, 0, msg="Lower should equal 0")
        self.assertEqual(p.upper, 1, msg="Upper should equal 1")
        self.assertEqual(p.size, 1, msg="Size of parameter should equal 1")

        p.upper = 2
        self.assertEqual(p.upper, 2, msg="After assignment, upper should equal 2")
        p.lower = 1
        self.assertEqual(p.lower, 1, msg="After assignment, lower should equal 2")

    def test_equality(self):
        p = GPflowOpt.domain.ContinuousParameter("x1", 0, 1)
        pne = GPflowOpt.domain.ContinuousParameter("x1", 0, 2)
        self.assertNotEqual(p, pne, msg="Should not be equal (invalid upper)")
        pne = GPflowOpt.domain.ContinuousParameter("x1", -1, 1)
        self.assertNotEqual(p, pne, msg="Should not be equal (invalid lower)")
        pne = GPflowOpt.domain.ContinuousParameter("x1", -1, 2)
        self.assertNotEqual(p, pne, msg="Should not be equal (invalid lower/upper)")
        p.lower = -1
        p.upper = 2
        self.assertEqual(p, pne, msg="Should be equal after adjusting bounds")

    def test_containment(self):
        p = GPflowOpt.domain.ContinuousParameter("x1", 0, 1)
        self.assertIn(0, p, msg="Point is within domain")
        self.assertIn(0.5, p, msg="Point is within domain")
        self.assertIn(1, p, msg="Point is within domain")
        self.assertNotIn(1.1, p, msg="Point is not within domain")
        self.assertNotIn(-0.5, p, msg="Point is not within domain")

    def test_value(self):
        p = GPflowOpt.domain.ContinuousParameter("x1", 0, 1)
        self.assertTupleEqual(p.value.shape, (1,), msg="Default value has incorrect shape.")
        self.assertTrue(np.allclose(p.value, 0.5), msg="Parameter has incorrect default value")

        p.value = 0.8
        self.assertTrue(np.allclose(p.value, 0.8), msg="Parameter has incorrect value after update")

        p.value = [0.6, 0.8]
        self.assertTupleEqual(p.value.shape, (2,), msg="Default value has incorrect shape.")
        np.testing.assert_allclose(p.value, np.array([0.6, 0.8]), err_msg="Parameter has incorrect value after update")

        p = GPflowOpt.domain.ContinuousParameter("x1", 0, 1, 0.2)
        self.assertTupleEqual(p.value.shape, (1,), msg="Default value has incorrect shape.")
        self.assertTrue(np.allclose(p.value, 0.2), msg="Parameter has incorrect initialized value")


class TestHypercubeDomain(unittest.TestCase):

    def setUp(self):
        self.domain = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1,4)])

    def test_object_integrity(self):
        self.assertEqual(len(self.domain._parameters), 3)

    def test_simple(self):
        self.assertEqual(self.domain.size, 3, msg="Size of domain should equal 3")
        self.assertTrue(np.allclose(self.domain.lower, -1.0), msg="Lower of domain should equal -1 for all parameters")
        self.assertTrue(np.allclose(self.domain.upper, 1.0), msg="Lower of domain should equal 1 for all parameters")

    def test_equality(self):
        dne = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1,3)] +
                      [GPflowOpt.domain.ContinuousParameter("x3", -3, 1)])
        self.assertNotEqual(self.domain, dne, msg="One lower bound mismatch, should not be equal.")
        dne = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                      [GPflowOpt.domain.ContinuousParameter("x3", -1, 2)])
        self.assertNotEqual(self.domain, dne, msg="One upper bound mismatch, should not be equal.")
        dne = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        self.assertNotEqual(self.domain, dne, msg="Size mismatch")
        de = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 4)])
        self.assertEqual(self.domain, de, msg="No mismatches, should be equal")

    def test_parenting(self):
        for p in self.domain:
            self.assertEqual(id(p._parent), id(self.domain), "Misspecified parent link detected")

    def test_access(self):
        for i in range(self.domain.size):
            self.assertEqual(self.domain[i].label, "x{0}".format(i+1), "Accessing parameters, encountering "
                                                                     "incorrect labels")

        self.domain[2].lower = -2
        de = np.sum([GPflowOpt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                     [GPflowOpt.domain.ContinuousParameter("x3", -2, 1)])

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
