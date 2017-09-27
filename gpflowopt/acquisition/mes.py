# Copyright 2017 Joachim van der Herten, Nicolas Knudde
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
from ..design import RandomDesign

from gpflow import settings
from gpflow.param import DataHolder
from gpflow.model import Model

import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect
import tensorflow as tf

float_type = settings.dtypes.float_type
stability = settings.numerics.jitter_level
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class MinValueEntropySearch(Acquisition):
    """
        Max-value entropy search acquisition function for single-objective global optimization.
        Introduced by (Wang et al., 2017).

        Key reference:

        ::
            @InProceedings{Wang:2017,
              title = 	 {Max-value Entropy Search for Efficient {B}ayesian Optimization},
              author = 	 {Zi Wang and Stefanie Jegelka},
              booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
              pages = 	 {3627--3635},
              year = 	 {2017},
              editor = 	 {Doina Precup and Yee Whye Teh},
              volume = 	 {70},
              series = 	 {Proceedings of Machine Learning Research},
              address = 	 {International Convention Centre, Sydney, Australia},
              month = 	 {06--11 Aug},
              publisher = 	 {PMLR},
            }
        """

    def __init__(self, model, domain, gridsize=10000, num_samples=10):
        assert isinstance(model, Model)
        super(MinValueEntropySearch, self).__init__(model)
        assert self.data[1].shape[1] == 1
        self.gridsize = gridsize
        self.num_samples = num_samples
        self.samples = DataHolder(np.zeros(num_samples, dtype=np_float_type))
        self._domain = domain

    def _setup(self):
        super(MinValueEntropySearch, self)._setup()

        # Apply Gumbel sampling
        m = self.models[0]
        valid = self.feasible_data_index()

        # Work with feasible data
        X = self.data[0][valid, :]
        N = np.shape(X)[0]
        Xrand = RandomDesign(self.gridsize, self._domain).generate()
        fmean, fvar = m.predict_f(np.vstack((X, Xrand)))
        idx = np.argmin(fmean[:N])
        right = fmean[idx].flatten()# + 2*np.sqrt(fvar[idx]).flatten()
        left = right
        probf = lambda x: np.exp(np.sum(norm.logcdf(-(x - fmean) / np.sqrt(fvar)), axis=0))

        i = 0
        while probf(left) < 0.75:
            left = 2. ** i * np.min(fmean - 5. * np.sqrt(fvar)) + (1. - 2. ** i) * right
            i += 1

        # Binary search for 3 percentiles
        q1, med, q2 = map(lambda val: bisect(lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.01),
                          [0.25, 0.5, 0.75])
        beta = (q1 - q2) / (np.log(np.log(4. / 3.)) - np.log(np.log(4.)))
        alpha = med + beta * np.log(np.log(2.))

        # obtain samples from y*
        mins = -np.log(-np.log(np.random.rand(self.num_samples).astype(np_float_type))) * beta + alpha
        self.samples.set_data(mins)

    def build_acquisition(self, Xcand):
        fmean, fvar = self.models[0].build_predict(Xcand)
        norm = tf.contrib.distributions.Normal(tf.constant(0.0, dtype=float_type), tf.constant(1.0, dtype=float_type))
        gamma = (fmean - tf.expand_dims(self.samples, axis=0)) / tf.sqrt(fvar)

        return tf.reduce_sum(gamma * norm.prob(gamma) / (2. * norm.cdf(gamma)) - norm.log_cdf(gamma),
                          axis=1, keep_dims=True) / self.num_samples
