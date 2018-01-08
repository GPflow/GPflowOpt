import gpflowopt
import numpy as np
import pytest


class TestContinuousParameter(object):

    @pytest.fixture()
    def p(self):
        yield gpflowopt.domain.ContinuousParameter("x1", 0, 1)

    def test_simple(self, p):
        np.testing.assert_allclose(p._range, [0, 1])
        assert p.lower == 0
        assert p.upper == 1
        assert p.size == 1
        p.upper = 2
        assert p.upper == 2
        p.lower = 1
        assert p.lower == 1
        p = np.sum([gpflowopt.domain.ContinuousParameter("x1", 0, 1)])
        assert p.size == 1

    def test_equality(self, p):
        pne = gpflowopt.domain.ContinuousParameter("x1", 0, 2)
        assert p != pne
        pne = gpflowopt.domain.ContinuousParameter("x1", -1, 1)
        assert p != pne
        pne = gpflowopt.domain.ContinuousParameter("x1", -1, 2)
        assert p != pne
        p.lower = -1
        p.upper = 2
        assert p == pne

    def test_indexing(self):
        p = np.sum([gpflowopt.domain.ContinuousParameter("x1", 0, 1),
                    gpflowopt.domain.ContinuousParameter("x2", 0, 1),
                    gpflowopt.domain.ContinuousParameter("x3", 0, 1),
                    gpflowopt.domain.ContinuousParameter("x4", 0, 1)])

        subdomain = p[['x4', 'x1', 2]]
        assert subdomain.size == 3
        assert subdomain[0].label == 'x4'
        assert subdomain[1].label == 'x1'
        assert subdomain[2].label == 'x3'

    def test_containment(self, p):
        assert 0 in p
        assert 0.5 in p
        assert 1 in p
        assert 1.1 not in p
        assert -0.5 not in p

    def test_value(self, p):
        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1)
        assert p.value.shape == (1,)
        np.testing.assert_allclose(p.value, 0.5)

        p.value = 0.8
        np.testing.assert_allclose(p.value, 0.8)

        p.value = [0.6, 0.8]
        assert p.value.shape == (2,)
        np.testing.assert_allclose(p.value, np.array([0.6, 0.8]))

        p = gpflowopt.domain.ContinuousParameter("x1", 0, 1, 0.2)
        assert p.value.shape == (1,)
        np.testing.assert_allclose(p.value, 0.2)


class TestHypercubeDomain(object):

    @pytest.fixture()
    def domain(self):
        yield np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 4)])

    def test_object_integrity(self, domain):
        assert len(domain._parameters) == 3

    def test_simple(self, domain):
        assert domain.size == 3
        np.testing.assert_allclose(domain.lower, -1.0)
        np.testing.assert_allclose(domain.upper, 1.0)

    def test_equality(self, domain):
        dne = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                     [gpflowopt.domain.ContinuousParameter("x3", -3, 1)])
        assert domain != dne
        dne = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                     [gpflowopt.domain.ContinuousParameter("x3", -1, 2)])
        assert domain != dne
        dne = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])
        assert domain != dne
        de = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 4)])
        assert domain == de

    def test_parenting(self, domain):
        for p in domain:
            assert id(p._parent) == id(domain)

    def test_access(self, domain):
        for i in range(domain.size):
            assert domain[i].label == "x{0}".format(i+1)

        domain[2].lower = -2
        de = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)] +
                    [gpflowopt.domain.ContinuousParameter("x3", -2, 1)])
        assert domain == de

    def test_containment(self, domain):
        A = np.random.rand(50,3)*2-1
        assert A in domain
        A = np.vstack((A, np.array([-2, -2, -2])))
        assert A not in domain
        A = np.random.rand(50,4)*2-1
        assert A not in domain

    def test_value(self, domain):
        assert domain.value.shape == (1, 3)
        np.testing.assert_allclose(domain.value, np.array([[0, 0, 0]]), err_msg="Parameter has incorrect initial value")

        A = np.random.rand(10, 3) * 2 - 1
        domain.value = A
        assert domain.value.shape == (10, 3)
        np.testing.assert_allclose(domain.value, A, err_msg="Parameter has incorrect value after assignment")

    def test_transformation(self, session, domain):
        X = np.random.rand(50,3)*2-1
        target = gpflowopt.domain.UnitCube(3)
        transform = domain >> target
        np.testing.assert_allclose(transform.forward(X), (X + 1) / 2)
        np.testing.assert_allclose(transform.backward(transform.forward(X)), X)

        inv_transform = target >> domain
        np.testing.assert_allclose(transform.backward(transform.forward(X)), inv_transform.forward(transform.forward(X)))
        np.testing.assert_allclose((~transform).A.value, inv_transform.A.value)
        np.testing.assert_allclose((~transform).b.value, inv_transform.b.value)

    def test_unitcube(self):
        domain = gpflowopt.domain.UnitCube(3)
        np.testing.assert_allclose(domain.lower, 0)
        np.testing.assert_allclose(domain.upper, 1)
        assert domain.size == 3

