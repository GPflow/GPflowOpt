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

from gpflow import settings

import numpy as np
import tensorflow as tf

float_type = settings.dtypes.float_type
stability = settings.numerics.jitter_level


class ProbabilityOfFeasibility(Acquisition):
    """
    Probability of Feasibility acquisition function for sampling feasible regions. Standard acquisition function for
    Bayesian Optimization with black-box expensive constraints.

    Key reference:
    
    ::
    
        @article{Schonlau:1997,
            title={Computer experiments and global optimization},
            author={Schonlau, Matthias},
            year={1997},
            publisher={University of Waterloo}
        }
       
    The acquisition function measures the probability of the latent function 
    being smaller than a threshold for a candidate point.

    .. math::
       \\alpha(\\mathbf x_{\\star}) = \\int_{-\\infty}^{0} \\, p(f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} ) \\, d f_{\\star}
    """

    def __init__(self, model, threshold=0.0, minimum_pof=0.5):
        """
        :param model: GPflow model (single output) representing our belief of the constraint
        :param threshold: Observed values lower than the threshold are considered valid
        :param minimum_pof: minimum pof score required for a point to be valid.
            For more information, see docstring of feasible_data_index
        """
        super(ProbabilityOfFeasibility, self).__init__(model)
        self.threshold = threshold
        self.minimum_pof = minimum_pof

    def constraint_indices(self):
        return np.arange(self.data[1].shape[1])

    def feasible_data_index(self):
        """
        Returns a boolean array indicating which points are feasible (True) and which are not (False).
        
        Answering the question *which points are feasible?* is slightly troublesome in case noise is present.
        Directly relying on the noisy data and comparing it to self.threshold does not make much sense.

        Instead, we rely on the model belief using the PoF (a probability between 0 and 1).
        As the implementation of the PoF corresponds to the cdf of the (normal) predictive distribution in
        a point evaluated at the threshold, requiring a minimum pof of 0.5 implies the mean of the predictive
        distribution is below the threshold, hence it is marked as feasible. A minimum pof of 0 marks all points valid.
        Setting it to 1 results in all invalid.
    
        :return: boolean ndarray (size N)
        """
        pred = self.evaluate(self.data[0])
        return pred.ravel() > self.minimum_pof

    def build_acquisition(self, Xcand):
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, stability)
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        return normal.cdf(tf.constant(self.threshold, dtype=float_type), name=self.__class__.__name__)
