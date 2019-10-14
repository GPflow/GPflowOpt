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

from abc import ABCMeta, abstractmethod

from gpflow.base import Module
from gpflow.models import BayesianModel

import numpy as np
import tensorflow as tf


class Acquisition(Module, metaclass=ABCMeta):
    """
    An acquisition function maps the belief represented by a Bayesian model into a
    score indicating how promising a point is for evaluation.

    In Bayesian Optimization this function is typically optimized over the optimization domain
    to determine the next point for evaluation. An object of this class holds a list of GPflow models. Subclasses
    implement a build_acquisition function which computes the acquisition function (usually from the predictive
    distribution) using TensorFlow. Optionally, a method setup can be implemented which computes some quantities which
    are used to compute the acquisition, but do not depend on candidate points.

    Acquisition functions can be combined through addition or multiplication to construct joint criteria. For instance,
    for constrained optimization. The objects then form a tree hierarchy.

    Acquisition models implement a lazy strategy to optimize models and run setup. This is implemented by a _needs_setup
    attribute (similar to the _needs_recompile in GPflow). Calling :meth:`set_data` sets this flag to True. Calling methods
    marked with the setup_require decorator (such as evaluate) optimize all models, then call setup if this flag is set.
    In hierarchies, first acquisition objects handling constraint objectives are set up, then the objects handling
    objectives.
    """

    def __init__(self, models=[]):
        """
        :param models: list of GPflow models representing our beliefs about the problem
        :param optimize_restarts: number of optimization restarts to use when training the models
        """
        super(Acquisition, self).__init__()
        self.models = np.atleast_1d(models).tolist()
        assert all(isinstance(model, BayesianModel) for model in self.models)

    def feasible_data_index(self, X):
        """
        Returns a boolean array indicating which data points are considered feasible (according to the acquisition
        function(s) ) and which not.
        
        By default all data is considered feasible.
        
        :return: logical indices to the feasible data points, size N
        """
        return tf.ones(tf.shape(X)[0], dtype=tf.bool)

    def setup(self, data):
        """
        Pre-calculation of quantities used later in the evaluation of the acquisition function for candidate points.
        
        Subclasses can implement this method to compute quantities (such as fmin). The decision when to run this function
        is governed by :class:`Acquisition`, based on the setup_required decorator on methods which require
        setup to be run (e.g. set_data).
        """
        pass


    @abstractmethod
    def evaluate(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, without returning the gradients.
        
        :return: acquisition scores, size N x 1
        """
        pass
