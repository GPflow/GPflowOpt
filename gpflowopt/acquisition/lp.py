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
from ..optim import SciPyOptimizer, MCOptimizer, StagedOptimizer

from GPflow import settings
from GPflow.param import DataHolder, AutoFlow

import tensorflow as tf
import numpy as np

stability = settings.numerics.jitter_level
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class LocalPenalization(Acquisition):

    def __init__(self, acquisition, domain, batch_size):
        super(LocalPenalization, self).__init__(batch_size=batch_size)
        assert (isinstance(acquisition, Acquisition))
        self.acq = acquisition
        self.M = DataHolder(np.zeros(1))
        self.L = DataHolder(np.zeros(1))
        self._domain = domain
        self._setup()

    @property
    def models(self):
        return self.acq.models if hasattr(self, 'acq') else []

    @AutoFlow((float_type, [None, None]))
    def gradient_norm(self, X):
        mu, _ = self.models[0]._build_predict(X)
        norm = tf.norm(tf.gradients(mu, [X])[0], axis=1)
        return norm, tf.gradients(norm, [X])[0]

    def _setup(self):
        super(LocalPenalization, self)._setup()
        opt = StagedOptimizer([MCOptimizer(self._domain, 5000), SciPyOptimizer(self._domain)])
        optimizer = SciPyOptimizer(self._domain)
        res = optimizer.optimize(lambda X: tuple(map(lambda y: -y, self.gradient_norm(X))))
        self.L.set_data(-res.fun)

        # Get M
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        samples_mean = self.models[0].predict_f(feasible_samples)[0]
        self.M.set_data(np.min(samples_mean, axis=0))

    def _penalizer(self, C, X):
        m, v = self.models[0]._build_predict(C)
        s = tf.sqrt(tf.maximum(v, stability)) / self.L
        r = (m - self.M) / self.L
        dists = tf.norm(X-C, axis=1, keep_dims=True)
        normal = tf.contrib.distributions.Normal(loc=np.array(0., dtype=np_float_type),
                                                 scale=np.array(1., dtype=np_float_type))
        return normal.log_cdf((dists - r) / s)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_penalty(self, C, X):
        return self._penalizer(C, X)

    def _recurse(self, Xs, Xp=[]):
        # Done, end recursion.
        if len(Xs) == 0:
            return 1.

        # Compute penalties if previous points are available
        if len(Xp) > 0:
            penalties = tf.concat(tuple(map(lambda C: self._penalizer(C, Xs[0]), Xp)), axis=1)
            penalty = tf.reduce_sum(penalties, axis=1, keep_dims=True)
        else:
            penalty = 0.

        # Acquisition score
        acq_score = tf.log(tf.nn.softplus(self.acq._build_acquisition(Xs[0])))

        # Score for this batch point
        scores = acq_score + penalty

        # Recurse and multiply
        return scores * self._recurse(Xs[1:], Xp + [Xs[0]])

    def _build_acquisition(self, *args):
        return self._recurse(args)

    def _optimize_models(self):
        if hasattr(self, 'acq'):
            self.acq._optimize_models()

