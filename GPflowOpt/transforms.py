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


from GPflow import settings
from GPflow.param import Parameterized, DataHolder, AutoFlow
import numpy as np
import tensorflow as tf

float_type = settings.dtypes.float_type


class DataTransform(Parameterized):
    """
    Implements a mapping from data in domain U to domain V.
    Useful for domain scaling.
    """

    @AutoFlow((float_type, [None, None]))
    def forward(self, X):
        """
        Performs numpy transformation of U -> V
        """
        return self.build_forward(X)

    def build_forward(self, X):
        """
        Performs Tensorflow transformation of U -> V
        :param X: N x P tensor
        :return: N x Q tensor
        """
        raise NotImplementedError

    def backward(self, Y):
        """
        Performs numpy transformation of V -> U. By default, calls forward on the inverted transform object which 
        requires implementation of __invert__. The method can be overwritten in subclasses if more efficient 
        implementation is available.
        :param Y: N x Q matrix
        :return: N x P matrix
        """
        return (~self).forward(Y)

    def build_backward(self, Y):
        """
        Performs numpy transformation of V -> U. By default, calls tf_forward on the inverted transform object which 
        requires implementation of __invert__. The method can be overwritten in subclasses if more efficient 
        implementation is available.
        :param Y: N x Q tensor
        :return: N x P tensor
                """
        return (~self).build_forward(Y)

    def __invert__(self):
        """
        Return a DataTransform object implementing the transform from V -> U
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LinearTransform(DataTransform):
    """
    Implements a simple linear transform of the form
    
    .. math::
       \\mathbf Y = (\\mathbf A \\mathbf X^{T})^{T} + \\mathbf b \\otimes \\mathbf 1_{N}^{T}

    """

    def __init__(self, A, b):
        """
        :param A: scaling matrix. Either a P-dimensional vector, or a P x P transformation matrix. For the latter, 
        the inverse and backward methods are not guaranteed to work as A must be invertible. It is also possible to 
        specify a matrix with size P x Q with Q != P to achieve a lower dimensional representation of X. In this case, 
        A is not invertible, hence inverse and backward are not supported.
        :param b: A P-dimensional offset vector.
        """
        super(LinearTransform, self).__init__()
        b = np.atleast_1d(b)
        A = np.atleast_1d(A)
        if len(A.shape) == 1:
            A = np.diag(A)

        assert (len(b.shape) == 1)
        assert (len(A.shape) == 2)

        self.A = DataHolder(A)
        self.b = DataHolder(b)

    @AutoFlow((float_type, [None, None]))
    def backward(self, Y):
        """
        Overwrites the default backward approach, it avoids an explicit matrix inversion.
        """
        return self.build_backward(Y)

    def build_forward(self, X):
        return tf.matmul(X, tf.transpose(self.A)) + self.b

    def build_backward(self, Y):
        """
        Overwrites the default backward approach, it avoids an explicit matrix inversion.
        """
        L = tf.cholesky(tf.transpose(self.A))
        XT = tf.matrix_triangular_solve(tf.transpose(L), tf.matrix_triangular_solve(L, tf.transpose(Y - self.b)))
        return tf.transpose(XT)

    def __invert__(self):
        A_inv = np.linalg.inv(self.A.value.T)
        return LinearTransform(A_inv, -np.dot(self.b.value, A_inv))

    def __str__(self):
        return 'XA + b'
