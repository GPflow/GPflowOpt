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

from gpflow.param import DataHolder
from gpflow import settings

import numpy as np
import tensorflow as tf

stability = settings.numerics.jitter_level


class ProbabilityOfImprovement(Acquisition):
    """
    Probability of Improvement acquisition function for single-objective global optimization.
    
    Key reference:
    
    ::
    
        @article{Kushner:1964,
            author = "Kushner, Harold J",
            journal = "Journal of Basic Engineering",
            number = "1",
            pages = "97--106",
            publisher = "American Society of Mechanical Engineers",
            title = "{A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise}",
            volume = "86",
            year = "1964"
        }

    .. math::
       \\alpha(\\mathbf x_{\\star}) = \\int_{-\\infty}^{f_{\\min}} \\, p( f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} ) \\, d f_{\\star}
    """

    def __init__(self, model):
        """
        :param model: GPflow model (single output) representing our belief of the objective 
        """
        super(ProbabilityOfImprovement, self).__init__(model)
        self.fmin = DataHolder(np.zeros(1))
        self._setup()

    def _setup(self):
        super(ProbabilityOfImprovement, self)._setup()
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        samples_mean, _ = self.models[0].predict_f(feasible_samples)
        self.fmin.set_data(np.min(samples_mean, axis=0))

    def build_acquisition(self, Xcand):
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, stability)
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        return normal.cdf(self.fmin, name=self.__class__.__name__)
