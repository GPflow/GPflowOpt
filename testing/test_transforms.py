import unittest
import GPflow
import GPflowOpt
import tensorflow as tf
from GPflow import settings
import numpy as np

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class DummyTransform(GPflowOpt.transforms.DataTransform):

    def __init__(self, c):
        self.value = c

    def forward(self, X):
        return X * self.value

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

    _multiprocess_can_split_ = True

    def setUp(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(float_type)
        self.y = tf.placeholder(float_type)

        self.x_np = np.random.rand(10, 2).astype(np_float_type)
        self.session = tf.Session()
        self.transforms = [DummyTransform(2.0), GPflowOpt.transforms.LinearTransform([2.0, 3.5], [1.2, 0.7])]

    def test_tf_np_forward(self):
        ys = [t.build_forward(self.x) for t in self.transforms]
        ys_tf = [self.session.run(y, feed_dict={self.x: self.x_np}) for y in ys]
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        for y1, y2 in zip(ys_tf, ys_np):
            self.assertTrue(np.allclose(y1, y2))

    def test_forward_backward(self):
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
        for x in xs_np:
            self.assertTrue(np.allclose(x, self.x_np))

    def test_tf_forward_backward(self):
        ys = [t.build_forward(self.x) for t in self.transforms]
        xs = [t.build_backward(y) for t, y in zip(self.transforms, ys)]
        xs_tf = [self.session.run(x, feed_dict={self.x: self.x_np}) for x in xs]
        for x in xs_tf:
            self.assertTrue(np.allclose(x, self.x_np))

    def test_invert_np(self):
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
        xsi_np = [(~t).forward(y) for t, y in zip(self.transforms, ys_np)]

        for x in zip(xs_np, xsi_np):
            self.assertTrue(np.allclose(x[0], self.x_np))
            self.assertTrue(np.allclose(x[1], self.x_np))
            self.assertTrue(np.allclose(x[0], x[1]))

    def test_invert_tf(self):
        ys = [t.build_forward(self.x) for t in self.transforms]
        xs = [t.build_backward(y) for t, y in zip(self.transforms, ys)]
        xsi = [(~t).build_forward(y) for t, y in zip(self.transforms, ys)]
        xs_tf = [self.session.run(x, feed_dict={self.x: self.x_np}) for x in zip(xs, xsi)]

        for x in xs_tf:
            self.assertTrue(np.allclose(x[0], self.x_np))
            self.assertTrue(np.allclose(x[1], self.x_np))
            self.assertTrue(np.allclose(x[0], x[1]))