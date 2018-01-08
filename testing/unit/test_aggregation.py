from .test_acquisition import SimpleAcquisition
import pytest
import gpflow
import gpflowopt
import numpy as np


@pytest.fixture(params=[gpflowopt.acquisition.AcquisitionSum, gpflowopt.acquisition.AcquisitionProduct])
def acquisition(request, parabola_model):
    cls = request.param
    yield cls([SimpleAcquisition(parabola_model), SimpleAcquisition(parabola_model)])


def test_object_integrity(acquisition):
    for oper in acquisition.operands:
        assert isinstance(oper, gpflowopt.acquisition.Acquisition)
    assert all(isinstance(m, gpflowopt.params.ModelWrapper) for m in acquisition.models)


def test_data(acquisition):
    np.testing.assert_allclose(acquisition.data[0], acquisition[0].data[0])
    np.testing.assert_allclose(acquisition.data[0], acquisition[1].data[0])
    Y = np.hstack(map(lambda model: model.Y.value, acquisition.models))
    np.testing.assert_allclose(acquisition.data[1], Y)


def test_enable_scaling(acquisition, domain):
    for oper in acquisition.operands:
        assert not any(m.wrapped.X.read_value() in gpflowopt.domain.UnitCube(2) for m in oper.models)

    acquisition.enable_scaling(domain)
    for oper in acquisition.operands:
        assert all(m.wrapped.X.read_value() in gpflowopt.domain.UnitCube(2) for m in oper.models)


def test_sum_validity(acquisition, domain):
    if not isinstance(acquisition, gpflowopt.acquisition.AcquisitionSum):
        pytest.skip()
    design = gpflowopt.design.FactorialDesign(4, domain)
    p1 = acquisition.evaluate(design.generate())
    p2 = acquisition.operands[0].evaluate(design.generate())
    np.testing.assert_allclose(p2, p1 / 2, rtol=1e-3)


def test_product_validity(acquisition, domain):
    if not isinstance(acquisition, gpflowopt.acquisition.AcquisitionProduct):
        pytest.skip()
    design = gpflowopt.design.FactorialDesign(4, domain)
    p1 = acquisition.evaluate(design.generate())
    p2 = acquisition.operands[0].evaluate(design.generate())
    np.testing.assert_allclose(p2, np.sqrt(p1), rtol=1e-3)


def test_indices(acquisition):
    np.testing.assert_allclose(acquisition.objective_indices(), np.arange(2, dtype=int))
    np.testing.assert_allclose(acquisition.constraint_indices(), np.arange(0, dtype=int))


def test_generating_operators(parabola_model):
    joint = gpflowopt.acquisition.ExpectedImprovement(parabola_model) + \
            gpflowopt.acquisition.ExpectedImprovement(parabola_model)
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionSum)

    joint = gpflowopt.acquisition.ExpectedImprovement(parabola_model) * \
            gpflowopt.acquisition.ExpectedImprovement(parabola_model)
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionProduct)


def test_multi_aggr(parabola_model):
    acq = [SimpleAcquisition(parabola_model) for _ in range(4)]
    acq1, acq2, acq3, acq4 = acq
    joint = acq1 + acq2 + acq3
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionSum)
    assert list(joint.operands.params) == [acq1, acq2, acq3]

    joint = acq1 * acq2 * acq3
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionProduct)
    assert list(joint.operands.params) == [acq1, acq2, acq3]

    first = acq2 + acq3
    assert isinstance(first, gpflowopt.acquisition.AcquisitionSum)
    assert list(first.operands.params) == [acq2, acq3]
    joint = acq1 + first
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionSum)
    assert list(joint.operands.params) == [acq1, acq2, acq3]

    first = acq2 * acq3
    assert isinstance(first, gpflowopt.acquisition.AcquisitionProduct)
    assert list(first.operands.params) == [acq2, acq3]
    joint = acq1 * first
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionProduct)
    assert list(joint.operands.params) == [acq1, acq2, acq3]

    first = acq1 + acq2
    second = acq3 + acq4
    joint = first + second
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionSum)
    assert list(joint.operands.params) == [acq1, acq2, acq3, acq4]

    first = acq1 * acq2
    second = acq3 * acq4
    joint = first * second
    assert isinstance(joint, gpflowopt.acquisition.AcquisitionProduct)
    assert list(joint.operands.params) == [acq1, acq2, acq3, acq4]


class SimpleConstraintAcquisition(SimpleAcquisition):
    def constraint_indices(self):
        return np.arange(1)


def test_hierarchy(parabola_model):
    joint = SimpleAcquisition(parabola_model) * \
            (SimpleConstraintAcquisition(parabola_model) + SimpleAcquisition(parabola_model))
    np.testing.assert_allclose(joint.objective_indices(), np.array([0, 2], dtype=int))
    np.testing.assert_allclose(joint.constraint_indices(), np.array([1], dtype=int))
