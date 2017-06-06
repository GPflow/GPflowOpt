import unittest
import GPflow
import GPflowOpt
import tensorflow as tf
from GPflow import settings
import numpy as np

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class DummyTransform(GPflowOpt.transforms.DataTransform):
    """
    As linear transform overrides backward/build_backward, create a different transform to obtain coverage of the
    default implementations
    """

    def __init__(self, c):
        super(DummyTransform, self).__init__()
        self.value = c

    def build_forward(self, X):
        return X * self.value

    def __invert__(self):
        return DummyTransform(1 / self.value)

    def __str__(self):
        return '(dummy)'


class LinearTransformTests(unittest.TestCase):
    """
    Tests are inspired on GPflow transform tests.
    """

    def setUp(self):
        self.x_np = np.random.rand(10, 2).astype(np_float_type)
        self.transforms = [DummyTransform(2.0), GPflowOpt.transforms.LinearTransform([2.0, 3.5], [1.2, 0.7])]

    def test_forward_backward(self):
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
        for x in xs_np:
            self.assertTrue(np.allclose(x, self.x_np))

    def test_invert_np(self):
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
        xsi_np = [(~t).forward(y) for t, y in zip(self.transforms, ys_np)]

        for x in zip(xs_np, xsi_np):
            self.assertTrue(np.allclose(x[0], self.x_np))
            self.assertTrue(np.allclose(x[1], self.x_np))
            self.assertTrue(np.allclose(x[0], x[1]))

    def test_backward_variance_full_cov(self):
        tf.reset_default_graph()
        t = ~GPflowOpt.transforms.LinearTransform([2.0, 1.0], [1.2, 0.7])
        x = tf.placeholder(float_type, [10, 10, 2])
        y = tf.placeholder(float_type, [None])
        t.make_tf_array(y)
        session = tf.Session()

        A = np.random.rand(10, 10)
        B1 = np.dot(A, A.T)
        A = np.random.rand(10, 10)
        B2 = np.dot(A, A.T)
        B = np.dstack((B1, B2))
        with t.tf_mode():
            scaled = t.build_backward_variance(x)
        feed_dict_keys = t.get_feed_dict_keys()
        feed_dict = {}
        t.update_feed_dict(feed_dict_keys, feed_dict)
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        feed_dict = {x: B, y: t.get_free_state()}
        t.update_feed_dict(feed_dict_keys, feed_dict)
        Bs = session.run(scaled, feed_dict=feed_dict)
        self.assertTrue(np.allclose(Bs[:, :, 0] / 4.0, B1))
        self.assertTrue(np.allclose(Bs[:, :, 1], B2))

    def test_backward_variance(self):
        tf.reset_default_graph()
        t = ~GPflowOpt.transforms.LinearTransform([2.0, 1.0], [1.2, 0.7])
        x = tf.placeholder(float_type, [10, 2])
        y = tf.placeholder(float_type, [None])
        t.make_tf_array(y)
        session = tf.Session()

        B = np.random.rand(10, 2)
        with t.tf_mode():
            scaled = t.build_backward_variance(x)
        feed_dict_keys = t.get_feed_dict_keys()
        feed_dict = {}
        t.update_feed_dict(feed_dict_keys, feed_dict)
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        feed_dict = {x: B, y: t.get_free_state()}
        t.update_feed_dict(feed_dict_keys, feed_dict)
        Bs = session.run(scaled, feed_dict=feed_dict)
        self.assertTrue(np.allclose(Bs, B * np.array([4, 1])))

    def test_assign(self):
        t1 = GPflowOpt.transforms.LinearTransform([2.0, 1.0], [1.2, 0.7])
        t2 = GPflowOpt.transforms.LinearTransform([1.0, 1.0], [0, 0])
        t1.assign(t2)
        np.testing.assert_allclose(t1.A.value, t2.A.value)
        np.testing.assert_allclose(t1.b.value, t2.b.value)
