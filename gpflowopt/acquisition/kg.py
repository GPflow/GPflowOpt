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

from .acquisition import Acquisition

from GPflow.model import Model
from GPflow import settings

import tensorflow as tf

stability = settings.numerics.jitter_level


class KGCP(Acquisition):

    def __init__(self, model):
        super(KGCP, self).__init__(model)
        assert (isinstance(model, Model))
        self._setup()

    def _setup(self):
        super(KGCP, self)._setup()

    def _build_acquisition(self, Xcand):
        N = tf.shape(self.models[0].X)[0]
        C = tf.shape[Xcand][0]
        sn2 = self.models[0].output_transform.build_backward_variance(self.models[0].likelihood.variance)
        # Obtain predictive distributions for candidates
        candidate_mean, candidate_var = self.models[0]._build_predict(Xcand, full_cov=True)
        a = -candidate_mean
        A = tf.concat((tf.tile(tf.slice(a, [0, 0], [N, 1]), (1, C)), tf.transpose(tf.slice(a, [N, 0], [C, 1]))), axis=0) # N+1 x C
        Q1 = tf.tile(tf.expand_dims(A, -1), (1, 1, C)) - tf.tile(tf.expand_dims(A, 1), (1, C, 1))  # N+1 x C x C

        candidate_var = tf.maximum(candidate_var, stability)
        v_samples = tf.slice(-candidate_var, [0, N], [N, C]) # N x C
        v_cand = tf.diag_part(tf.slice(-candidate_var, [N, N], [C, C])) # C
        B = tf.concat((v_samples, tf.expand_dims(v_cand, 0)), axis=0) / tf.sqrt(sn2 + v_cand) # N+1 x C
        Q2 = tf.tile(tf.expand_dims(B, -1), (1, 1, C)) - tf.tile(tf.expand_dims(B, 1), (1, C, 1)) # N+1 x C x C

        C = Q1 / Q2



