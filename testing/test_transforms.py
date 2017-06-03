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
        return DummyTransform(1/self.value)

    def __str__(self):
        return '(dummy)'


class LinearTransformTests(unittest.TestCase):
    """
    Tests are inspired on GPflow transform tests.
    """

    def setUp(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(float_type)
        self.y = tf.placeholder(float_type)

        self.x_np = np.random.rand(10, 2).astype(np_float_type)
        self.session = tf.Session()
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