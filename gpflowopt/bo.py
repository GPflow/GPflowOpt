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

from abc import ABCMeta

import numpy as np
import tensorflow as tf
from scipy.optimize import OptimizeResult

from .acquisition import Acquisition
from .experiment import Experiment
from .domain import Domain
from .factory import ModelFactory
from .objective import ObjectiveWrapper
from .optim import SciPyOptimizer, Optimizer


class BayesianOptimizer(Optimizer, metaclass=ABCMeta):
    """
    A traditional Bayesian optimization framework implementation.
    
    Like other optimizers, this optimizer is constructed for optimization over a domain.
    Additionally, it is configured with a separate optimizer for the acquisition function.
    """

    def __init__(self, data: Experiment, model_factory: ModelFactory, acquisition: Acquisition):
        """
        :param Domain domain: The optimization space.
        :param Acquisition acquisition: The acquisition function to optimize over the domain.
        """
        assert isinstance(acquisition, Acquisition)
        super(BayesianOptimizer, self).__init__(data.domain, exclude_gradient=True)
        self.acquisition = acquisition
        self.acquisition_optimizer = SciPyOptimizer(data.domain)
        self.model_factory = model_factory
        self.data = data

    def _update_models(self) -> None:
        X, Y = self.data
        update = []
        for m, y in zip(self.acquisition.models, Y.T):
            slice = (X, y[:, None])
            update.append(self.model_factory.update_model(m, slice))
        self.acquisition.models = update

    def _optimize_acquisition(self):
        def inverse_acquisition(x):
            Xcand = np.atleast_2d(x)
            acq = self.acquisition.evaluate(Xcand)
            grad = tf.gradients(acq, [Xcand])[0]
            return -acq.numpy(), -grad.numpy()

        result = self.acquisition_optimizer.optimize(inverse_acquisition)
        return result.x

    def _iteration(self):
        self._update_models()
        self.acquisition.setup(self.data)
        return self._opimize_acquisition()

    def _optimize(self, fx: ObjectiveWrapper, n_iter: int) -> OptimizeResult:
        """
        Internal optimization function. Receives an ObjectiveWrapper as input. As exclude_gradient is set to true,
        the placeholder created by :meth:`_evaluate_objectives` will not be returned.
       
        :param fx: :class:`.objective.ObjectiveWrapper` object wrapping expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """
        assert isinstance(fx, ObjectiveWrapper)

        # Optimization loop
        for i in range(n_iter):
            Xnew = self._iteration()
            Ynew = fx(Xnew)
            self.data.add((Xnew, Ynew))

        return "OK"

