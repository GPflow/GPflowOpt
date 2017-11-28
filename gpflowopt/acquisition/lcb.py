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
import numpy as np

import tensorflow as tf


class LowerConfidenceBound(Acquisition):
    """
    Lower confidence bound acquisition function for single-objective global optimization.
    
    Key reference:
    
    ::
    
        @inproceedings{Srinivas:2010,
            author = "Srinivas, Niranjan and Krause, Andreas and Seeger, Matthias and Kakade, Sham M.",
            booktitle = "{Proceedings of the 27th International Conference on Machine Learning (ICML-10)}",
            editor = "F{\"u}rnkranz, Johannes and Joachims, Thorsten",
            pages = "1015--1022",
            publisher = "Omnipress",
            title = "{Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design}",
            year = "2010"
        }

    .. math::
       \\alpha(\\mathbf x_{\\star}) =\\mathbb{E} \\left[ f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} \\right]
       - \\sigma \\mbox{Var} \\left[ f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} \\right]
    """

    def __init__(self, model, sigma=2.0):
        """
        :param model: GPflow model (single output) representing our belief of the objective 
        :param sigma: See formula, the higher the more exploration
        """
        super(LowerConfidenceBound, self).__init__(model)
        self.sigma = DataHolder(np.array(sigma))

    def build_acquisition(self, Xcand):
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, 0)
        return tf.subtract(candidate_mean, self.sigma * tf.sqrt(candidate_var), name=self.__class__.__name__)
