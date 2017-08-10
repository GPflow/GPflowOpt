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

from GPflow import settings
from GPflow.param import DataHolder

import numpy as np
from scipy.stats import norm
import tensorflow as tf

float_type = settings.dtypes.float_type
stability = settings.numerics.jitter_level


class MaxvalueEntropySearch(Acquisition):
    def __init__(self, model, domain, gridsize=10000, num_samples=10):
        super(MaxvalueEntropySearch, self).__init__(model, domain=domain)
        self.gridsize = gridsize
        self.num_samples = num_samples
        self.samples = DataHolder(np.zeros(num_samples))

    def setup(self):
        super(MaxvalueEntropySearch, self).setup()
        m = self.models[0].wrapped
        X = np.array(m.X.value.copy())
        D = np.shape(X)[1]

        Xrand = np.random.rand(self.gridsize, D)
        fmean, fvar = m.predict_f(np.concatenate((Xrand, X), axis=0))

        right = np.max(m.Y.value.copy())

        probf = lambda x: np.exp(np.sum(norm.logcdf(-(x - fmean) / np.sqrt(fvar))))

        left = np.min(fmean - 5 * np.sqrt(fvar))
        while probf(left) < 0.75:
            left = -2 * left + right

        q1 = self.binary_search(left, right, probf, 0.25)
        q2 = self.binary_search(left, right, probf, 0.75)
        med = self.binary_search(left, right, probf, 0.5)
        beta = (q1 - q2) / (np.log(np.log(4 / 3)) - np.log(np.log(4)))
        alpha = med + beta * np.log(np.log(2))
        mins = - np.log(-np.log(np.random.rand(self.num_samples))) * beta + alpha
        m._kill_autoflow()
        self.samples.set_data(mins)

    def binary_search(self, left, right, func, val, threshold=0.01):
        vfunc = np.vectorize(func)
        x = np.flip(np.linspace(left, right, 100), axis=0)
        y = vfunc(x)
        i = np.searchsorted(y, val)
        l, r = x[i], x[i - 1]
        mid = (l + r) / 2
        ev = func(mid)
        while np.abs(ev - val) > threshold:
            if ev > val:
                l = mid
                r = r
            else:
                l = l
                r = mid
            mid = (l + r) / 2
            ev = func(mid)

        return mid

    def build_acquisition(self, Xcand):
        fmean, fvar = self.models[0].wrapped.build_predict(Xcand)
        norm = tf.contrib.distributions.Normal(loc=tf.zeros([], dtype=float_type), scale=tf.ones([], dtype=float_type))

        gamma = (fmean - tf.expand_dims(self.samples, axis=0)) / tf.sqrt(fvar)

        a = tf.reduce_sum(gamma * norm.prob(gamma) / (2 * norm.cdf(gamma)) - norm.log_cdf(gamma),
                          axis=1, keep_dims=True) / self.num_samples

        return a
