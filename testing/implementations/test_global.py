import gpflowopt
import pytest
import numpy as np


acquisitions = [cls for cls in gpflowopt.acquisition.Acquisition.__subclasses__()
                if not cls == gpflowopt.acquisition.AcquisitionAggregation]


@pytest.fixture(params=acquisitions)
def acquisition(request, parabola_model, domain):
    if request.param == gpflowopt.acquisition.HVProbabilityOfImprovement:
        yield request.param([parabola_model, parabola_model])
    elif request.param == gpflowopt.acquisition.MinValueEntropySearch:
        yield request.param(parabola_model, domain)
    else:
        yield request.param(parabola_model)


def test_acquisition_evaluate(acquisition, domain):
    X = gpflowopt.design.RandomDesign(10, domain).generate()
    p = acquisition.evaluate(X)
    assert isinstance(p, np.ndarray)
    assert p.shape == (10, 1)

    q = acquisition.evaluate_with_gradients(X)
    assert isinstance(q, tuple)
    assert len(q) == 2
    assert all(isinstance(q[i], np.ndarray) for i in range(2))
    assert q[0].shape == (10, 1)
    assert q[1].shape == (10, 2)
    np.testing.assert_allclose(p, q[0])