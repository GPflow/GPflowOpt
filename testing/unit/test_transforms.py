import gpflowopt
import tensorflow as tf
from gpflow import settings
import numpy as np
import pytest

float_type = settings.tf_float
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class DummyTransform(gpflowopt.transforms.DataTransform):
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


transforms = [
    (DummyTransform, (2.0,)),
    (gpflowopt.transforms.LinearTransform, ([2.0, 3.5], [1.2, 0.7]))
]


@pytest.mark.parametrize('transform,args', transforms)
def test_forward_backward(transform, args):
    x_np = np.random.rand(10, 2).astype(np_float_type)
    with tf.Session(graph=tf.Graph()):
        t = transform(*args)
        y = t.forward(x_np)
        x = t.backward(y)
        np.testing.assert_allclose(x, x_np)


@pytest.mark.parametrize('transform,args', transforms)
def test_invert_np(transform, args):
    x_np = np.random.rand(10, 2).astype(np_float_type)
    with tf.Session(graph=tf.Graph()):
        t = transform(*args)
        y = t.forward(x_np)
        x = t.backward(y)
        xi = (~t).forward(y)
        np.testing.assert_allclose(x, x_np)
        np.testing.assert_allclose(xi, x_np)
        np.testing.assert_allclose(x, xi)


def test_backward_variance_full_cov():
    with tf.Session(graph=tf.Graph()) as session:
        t = ~gpflowopt.transforms.LinearTransform([2.0, 1.0], [1.2, 0.7])
        x = tf.placeholder(float_type, [10, 10, 2])

        A = np.random.rand(10, 10)
        B1 = np.dot(A, A.T)
        A = np.random.rand(10, 10)
        B2 = np.dot(A, A.T)
        B = np.dstack((B1, B2))
        scaled = t.build_backward_variance(x)
        Bs = session.run(scaled, feed_dict={x: B})

        np.testing.assert_allclose(Bs[:, :, 0] / 4.0, B1)
        np.testing.assert_allclose(Bs[:, :, 1], B2)


def test_backward_variance():
    with tf.Session(graph=tf.Graph()) as session:
        t = ~gpflowopt.transforms.LinearTransform([2.0, 1.0], [1.2, 0.7])
        x = tf.placeholder(float_type, [10, 2])
        
        B = np.random.rand(10, 2)
        scaled = t.build_backward_variance(x)
        Bs = session.run(scaled, feed_dict={x: B})
        np.testing.assert_allclose(Bs, B * np.array([4, 1]))


def test_assign():
    with tf.Session(graph=tf.Graph()):
        t1 = gpflowopt.transforms.LinearTransform([2.0, 1.0], [1.2, 0.7])
        t2 = gpflowopt.transforms.LinearTransform([1.0, 1.0], [0., 0.])
        t1.assign(t2)
        np.testing.assert_allclose(t1.A.read_value(), t2.A.read_value())
        np.testing.assert_allclose(t1.b.read_value(), t2.b.read_value())

