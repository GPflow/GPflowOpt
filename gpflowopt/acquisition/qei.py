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

from .acquisition import ParallelBatchAcquisition

from gpflow.model import Model
from gpflow.param import DataHolder
from gpflow import settings

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds

stability = settings.numerics.jitter_level
float_type = settings.dtypes.float_type


class qExpectedImprovement(ParallelBatchAcquisition):
    def __init__(self, model, batch_size=4):
        """
        :param model: GPflow model (single output) representing our belief of the objective
        """
        super(qExpectedImprovement, self).__init__(model, batch_size=batch_size)
        self.fmin = DataHolder(np.zeros(1))
        self._setup()

    def _setup(self):
        super(qExpectedImprovement, self)._setup()
        # Obtain the lowest posterior mean for the previous - feasible - evaluations
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        samples_mean, _ = self.models[0].predict_f(feasible_samples)
        self.fmin.set_data(np.min(samples_mean, axis=0))

    def build_acquisition(self, *args):
        # Obtain predictive distributions for candidates
        N, D = tf.shape(args[0])[0], tf.shape(args[0])[1]
        q = self.batch_size
        #Q x N x D
        Xcand = tf.transpose(tf.stack(args, axis=0), perm=[1, 0, 2]) # N x Q x D
        m, sig = tf.map_fn(lambda x: self.models[0].build_predict(x, full_cov=True), Xcand,
                           dtype=(float_type, float_type))  # N x q x 1, N x q x q

        eye = tf.tile(tf.expand_dims(tf.eye(q, dtype=float_type), 0), [q, 1, 1])
        A = eye
        A = A - tf.transpose(eye, perm=[1, 2, 0])
        A = A - tf.transpose(eye, perm=[2, 0, 1])  # q x q x q     (q(k) x q x q)

        mk = tf.squeeze(tf.tensordot(m, A, [[1], [2]]), 1)  # N x q(k) x q  Mean of Zk   (k x q)
        mk = tf.reshape(mk, [N, q, q])
        sigk = tf.transpose(tf.tensordot(sig, A, [[1], [2]]), [0, 2, 3, 1])  # N x q(k) x q x q
        sigk = tf.reduce_sum(tf.expand_dims(sigk, 3) * tf.expand_dims(tf.expand_dims(A, 0), 2), axis=-1)# N x q(k) x q x q
        sigk = tf.reshape(sigk, [N, q, q, q])

        a = tf.tile(tf.expand_dims(tf.eye(q, q, dtype=float_type), 0), [q, 1, 1])
        A1 = tf.gather_nd(a, [[[i, j + (j >= i)] for j in range(q)] for i in range(q-1)])
        # q(i) x (q-1) x q

        bk = -self.fmin * tf.eye(q, dtype=float_type)  # q(k) x q

        Sigk_t = tf.expand_dims(tf.transpose(tf.tensordot(sigk, A1, [[-2], [2]]), [0, 1, 3, 4, 2]), -2)  # N x q(k) x q(i) x (q-1) x 1 x q
        Sigk = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.expand_dims(A1, 0), 0), -3)* Sigk_t, axis=-1)
        #Sigk = tf.einsum('ijklm,lnk->ijlmn', Sigk_t, A1)  # N x q(k) x q(i) x q-1 x q-1
        c = tf.tensordot(tf.expand_dims(bk, 0) - mk, A1, [[2], [2]])  # N x q(k) x q(i) x q-1
        F = tf.expand_dims((tf.expand_dims(bk, 0) - mk) / tf.matrix_diag_part(sigk),
                           -1)  # N x q(k) x q(i) x 1
        F *= tf.transpose(tf.squeeze(tf.matrix_diag_part(tf.transpose(Sigk_t, [0, 1, 3, 4, 5, 2])), 3), [0, 1, 3, 2])
        c -= F

        bk = tf.tile(tf.expand_dims(bk, 0), [N, 1, 1])

        MVN = ds.MultivariateNormalFullCovariance(loc=mk, covariance_matrix=sigk)
        MVN2 = ds.MultivariateNormalFullCovariance(loc=tf.zeros(tf.shape(c), dtype=float_type), covariance_matrix=Sigk)
        MVN._is_maybe_event_override = False
        MVN2._is_maybe_event_override = False


        UVN = ds.MultivariateNormalDiag(loc=mk, scale_diag=tf.sqrt(tf.matrix_diag_part(sigk)))
        t1 = tf.reduce_sum((self.fmin - m) * MVN.cdf(bk), axis=1)
        sigkk = tf.transpose(tf.matrix_diag_part(tf.transpose(sigk, perm=[0, 3, 1, 2])), perm=[0, 2, 1])
        t2 = tf.reduce_sum(sigkk * UVN.prob(-tf.expand_dims(bk, 0)) * MVN2.cdf(c), axis=[1, 2])
        return tf.add(t1, t2, name=self.__class__.__name__)
