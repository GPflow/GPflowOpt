import pytest
import gpflowopt
import numpy as np


@pytest.fixture()
def lcb(plane_model):
    yield gpflowopt.acquisition.LowerConfidenceBound(plane_model, 3.2)


def test_objective_indices(lcb):
    assert lcb.objective_indices() == np.arange(1, dtype=int)


def test_object_integrity(lcb):
    np.testing.assert_allclose(lcb.sigma.read_value(), 3.2)


def test_lcb_validity(domain, lcb):
    design = gpflowopt.design.RandomDesign(200, domain).generate()
    q = lcb.evaluate(design)
    p = lcb.models[0].predict_f(design)[0]
    np.testing.assert_array_less(q, p)


def test_lcb_validity_2(domain, lcb):
    design = gpflowopt.design.RandomDesign(200, domain).generate()
    lcb.sigma = 0
    q = lcb.evaluate(design)
    p = lcb.models[0].predict_f(design)[0]
    np.testing.assert_allclose(q, p)