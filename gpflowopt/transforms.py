# Copyright 2017 Joachim van der Herten
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gpflow import Parameterized, DataHolder, autoflow, settings, params_as_tensors
import numpy as np
import tensorflow as tf


class DataTransform(Parameterized):
    """
    Maps data in :class:`.Domain` U to :class:`.Domain` V.

    Useful for scaling of data between domains.
    """

    @autoflow((settings.tf_float, [None, None]))
    def forward(self, X):
        """
        Performs the transformation of U -> V
        """
        return self.build_forward(X)

    def build_forward(self, X):
        """
        Tensorflow graph for the transformation of U -> V

        :param X: N x P tensor
        :return: N x Q tensor
        """
        raise NotImplementedError

    def backward(self, Y):
        """
        Performs the transformation of V -> U. By default, calls the :meth:`.forward` transform on the inverted
        transform object which requires implementation of __invert__. The method can be overwritten in subclasses if a
        more efficient (direct) transformation is possible.

        :param Y: N x Q matrix
        :return: N x P matrix
        """
        return (~self).forward(Y)

    def assign(self, other):
        raise NotImplementedError

    def __invert__(self):
        """
        Return a :class:`.DataTransform` object implementing the reverse transform V -> U
        """
        raise NotImplementedError


class LinearTransform(DataTransform):
    """
    A simple linear transform of the form
    
    .. math::
       \\mathbf Y = (\\mathbf A \\mathbf X^{T})^{T} + \\mathbf b \\otimes \\mathbf 1_{N}^{T}

    """

    def __init__(self, A, b):
        """
        :param A: scaling matrix. Either a P-dimensional vector, or a P x P transformation matrix. For the latter, 
            the inverse and backward methods are not guaranteed to work as A must be invertible.
            
            It is also possible to specify a matrix with size P x Q with Q != P to achieve 
            a lower dimensional representation of X.
            In this case, A is not invertible, hence inverse and backward transforms are not supported.
        :param b: A P-dimensional offset vector.
        """
        super(LinearTransform, self).__init__()
        assert A is not None
        assert b is not None

        b = np.atleast_1d(b)
        A = np.atleast_1d(A)
        if len(A.shape) == 1:
            A = np.diag(A)

        assert (len(b.shape) == 1)
        assert (len(A.shape) == 2)

        self.A = DataHolder(A)
        self.b = DataHolder(b)

    @params_as_tensors
    def build_forward(self, X):
        return tf.matmul(X, tf.transpose(self.A)) + self.b

    @autoflow((settings.tf_float, [None, None]))
    def backward(self, Y):
        """
        Overwrites the default backward approach, to avoid an explicit matrix inversion.
        """
        return self.build_backward(Y)

    @params_as_tensors
    def build_backward(self, Y):
        """
        TensorFlow implementation of the inverse mapping
        """
        L = tf.cholesky(tf.transpose(self.A))
        XT = tf.cholesky_solve(L, tf.transpose(Y-self.b))
        return tf.transpose(XT)

    @params_as_tensors
    def build_backward_variance(self, Yvar):
        """
        Additional method for scaling variance backward (used in :class:`.Normalizer`). Can process both the diagonal
        variances returned by predict_f, as well as full covariance matrices.

        :param Yvar: size P x N x N or size N x P
        :return: Yvar scaled, same rank and size as input
        """
        rank = tf.rank(Yvar)

        # Because TensorFlow evaluates both fn1 and fn2, the transpose can't be in the same line. If a full cov
        # matrix is provided fn1 turns it into a rank 4, then tries to transpose it as a rank 3.
        # Splitting it in two steps however works fine.
        Yvar = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.matrix_diag(tf.transpose(Yvar)),
            lambda: Yvar
        )
        Yvar = tf.transpose(Yvar, perm=[1, 2, 0])

        N = tf.shape(Yvar)[0]
        D = tf.shape(Yvar)[2]
        L = tf.cholesky(tf.square(tf.transpose(self.A)))
        Yvar = tf.reshape(Yvar, [N * N, D])
        scaled_var = tf.reshape(tf.transpose(tf.cholesky_solve(L, tf.transpose(Yvar))), [N, N, D])
        return tf.cond(
            tf.equal(rank, 2),
            lambda: tf.reduce_sum(scaled_var, axis=1),
            lambda: tf.transpose(scaled_var, perm=[2, 0, 1])
        )

    def assign(self, other):
        """
        Assign the parameters of another :class:`LinearTransform`.

        Useful to avoid graph re-compilation.

        :param other: :class:`.LinearTransform` object
        """
        assert other is not None
        assert isinstance(other, LinearTransform)
        self.A = other.A.read_value()
        self.b = other.b.read_value()

    def __invert__(self):
        A_inv = np.linalg.inv(self.A.read_value().T)
        return LinearTransform(A_inv, -np.dot(self.b.read_value(), A_inv))

