import pytest
import gpflowopt
import numpy as np


@pytest.fixture()
def hvpoi(vlmop2_models):
    yield gpflowopt.acquisition.HVProbabilityOfImprovement(vlmop2_models)


def test_object_integrity(hvpoi, vlmop2_models):
    assert len(hvpoi.models) == 2
    for m1, m2 in zip(hvpoi.models, vlmop2_models):
        assert m1 == m2


def test_hvpoi_validity(hvpoi, vlmop2_data):
    scores = hvpoi.evaluate(vlmop2_data['candidates'])
    np.testing.assert_almost_equal(scores, vlmop2_data['scores'], decimal=2)