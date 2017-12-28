import gpflowopt
import numpy as np
import gpflow
import tensorflow as tf
import pytest
from gpflow.test_util import GPflowTestCase
from ..utility import create_parabola_model, parabola2d, plane
import contextlib
import copy

domain = np.sum([gpflowopt.domain.ContinuousParameter("x{0}".format(i), -1, 1) for i in range(1, 3)])


class SimpleAcquisition(gpflowopt.acquisition.Acquisition):
    def __init__(self, model):
        super(SimpleAcquisition, self).__init__(model)
        self.counter = 0

    def _setup(self):
        super(SimpleAcquisition, self)._setup()
        self.counter += 1

    def _build_acquisition(self, Xcand):
        return self.models[0]._build_predict(Xcand)[0]


class TestAcquisition(GPflowTestCase):

    def setUp(self):
        self.model = create_parabola_model(domain)
        self.acquisition = SimpleAcquisition(self.model)

    def get_acq_state(self):
        return copy.deepcopy(self.acquisition.read_trainables())

    def compare_model_state(self, sa, sb):
        eq_keys = list(sa.keys()) == list(sb.keys())
        a = np.hstack(v for v in sa.values())
        b = np.hstack(v for v in sb.values())
        return eq_keys and np.allclose(a, b)

    @contextlib.contextmanager
    def test_context(self):
        yield super(TestAcquisition, self).test_context(graph=self.model.graph)

    def test_object_integrity(self):
        self.assertEqual(len(self.acquisition.models), 1, msg="Model list has incorrect length.")
        self.assertEqual(self.acquisition.models[0], self.model, msg="Incorrect model stored.")

    def test_setup_trigger(self):
        with self.test_context():
            m = create_parabola_model(domain)
            init_state = dict(zip(self.acquisition.models[0].read_trainables().keys(), m.read_trainables().values()))
            self.assertTrue(self.compare_model_state(init_state, self.get_acq_state()))
            self.assertTrue(self.acquisition._needs_setup)
            self.assertEqual(self.acquisition.counter, 0)
            self.acquisition.evaluate(gpflowopt.design.RandomDesign(10, domain).generate())
            self.assertFalse(self.acquisition._needs_setup)
            self.assertEqual(self.acquisition.counter, 1)
            self.assertFalse(self.compare_model_state(init_state, self.get_acq_state()))

            self.acquisition._needs_setup = True
            self.acquisition.models[0].assign(init_state)
            self.acquisition.evaluate_with_gradients(gpflowopt.design.RandomDesign(10, domain).generate())
            self.assertFalse(self.acquisition._needs_setup)
            self.assertEqual(self.acquisition.counter, 2)

    def test_data(self):
        # Test the data property
       with self.test_context():
           with gpflow.params_as_tensors_for(self.acquisition):
               self.assertTrue(isinstance(self.acquisition.data[0], tf.Tensor))
               self.assertTrue(isinstance(self.acquisition.data[1], tf.Tensor))

    def test_data_update(self):
        # Verify the effect of setting the data
        with self.test_context():
            design = gpflowopt.design.RandomDesign(10, domain)
            X = np.vstack((self.acquisition.data[0], design.generate()))
            Y = parabola2d(X)
            self.acquisition._needs_setup = False
            self.acquisition.set_data(X, Y)
            np.testing.assert_allclose(self.acquisition.data[0], X, atol=1e-5, err_msg="Samples not updated")
            np.testing.assert_allclose(self.acquisition.data[1], Y, atol=1e-5, err_msg="Values not updated")
            self.assertTrue(self.acquisition._needs_setup)

    def test_data_indices(self):
        # Return all data as feasible.
        with self.test_context():
            self.assertTupleEqual(self.acquisition.feasible_data_index().shape, (self.acquisition.data[0].shape[0],))

    def test_enable_scaling(self):
        with self.test_context():
            self.assertFalse(
                any(m.wrapped.X.read_value() in gpflowopt.domain.UnitCube(domain.size) for m in self.acquisition.models))
            self.acquisition._needs_setup = False
            self.acquisition.enable_scaling(domain)
            self.assertTrue(
                all(m.wrapped.X.read_value() in gpflowopt.domain.UnitCube(domain.size) for m in self.acquisition.models))
            self.assertTrue(self.acquisition._needs_setup)

    def test_result_shape_tf(self):
        # Verify the returned shape of evaluate
        with self.test_context():
            x_tf = tf.placeholder(tf.float64, shape=(50, 2))
            tens = self.acquisition._build_acquisition(x_tf)
            self.assertTrue(isinstance(tens, tf.Tensor), msg="no Tensor was returned")

    def test_result_shape_np(self):
        with self.test_context():
            design = gpflowopt.design.RandomDesign(50, domain)
            res = self.acquisition.evaluate(design.generate())
            self.assertTupleEqual(res.shape, (50, 1))
            res = self.acquisition.evaluate_with_gradients(design.generate())
            self.assertTrue(isinstance(res, tuple))
            self.assertTrue(len(res), 2)
            self.assertTupleEqual(res[0].shape, (50, 1))
            self.assertTupleEqual(res[1].shape, (50, domain.size))

    def test_optimize(self):
        with self.test_context():
            self.acquisition.optimize_restarts = 0
            state = self.get_acq_state()
            self.acquisition._optimize_models()
            self.assertTrue(self.compare_model_state(state, self.get_acq_state()))

            self.acquisition.optimize_restarts = 1
            self.acquisition._optimize_models()
            self.assertFalse(self.compare_model_state(state, self.get_acq_state()))


