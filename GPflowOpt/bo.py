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

import numpy as np
from scipy.optimize import OptimizeResult

from .acquisition import Acquisition
from .optim import Optimizer, SciPyOptimizer
from .design import EmptyDesign


class BayesianOptimizer(Optimizer):
    """
    Specific implementation of the Optimizer interface representing bayesian optimization.
    Takes a domain, an acquisition function and a seperate, second optimizer object for the acquisition function.
    """

    def __init__(self, domain, acquisition, optimizer=None, initial=None):
        assert isinstance(acquisition, Acquisition)
        super(BayesianOptimizer, self).__init__(domain, exclude_gradient=True)
        self.acquisition = acquisition
        if initial is None:
            initial = EmptyDesign(domain)
        self.set_initial(initial.generate())
        self.optimizer = SciPyOptimizer(domain) if optimizer is None else optimizer

    def _update_model_data(self, newX, newY):
        """
        Update the underlying models of the acquisition function with new data
        :param newX: samples (# new samples x indim)
        :param newY: values obtained by evaluating the objective and constraint functions (# new samples x # targets)
        """
        assert self.acquisition.data[0].shape[1] == newX.shape[-1]
        assert self.acquisition.data[1].shape[1] == newY.shape[-1]
        assert newX.shape[0] == newY.shape[0]
        X = np.vstack((self.acquisition.data[0], newX))
        Y = np.vstack((self.acquisition.data[1], newY))
        self.acquisition.set_data(X, Y)

    def _evaluate_objectives(self, X, fxs):
        """
        Evaluates a list of n functions on X. Returns a ndarray, with the number of columns equal to sum(Q0,...Qn-1)
        with Qi the number of columns obtained by evaluating the i-th function.
        :param X: input points, 2D ndarray, N x D
        :param fxs: 1D ndarray of (expensive) functions
        :return: tuple: (0) 2D ndarray (# new samples x sum(Q0,...Qn-1)). Evaluations
                        (1) 2D ndarray (# new samples x 0): Bayesian Optimizer is gradient-free, however calling
                        optimizer of the parent class expects a gradient. Will be discarded further on.
        """
        if X.size > 0:
            evaluations = np.hstack(map(lambda f: f(X), fxs))
            assert evaluations.shape[1] == self.acquisition.data[1].shape[1]
            return evaluations, np.zeros((X.shape[0], 0))
        else:
            return np.empty((0, self.acquisition.data[1].shape[1])), np.zeros((0, 0))

    def _create_bo_result(self, success, message):
        """
        Given all data evaluated after the optimization, analyse and return an OptimizeResult
        :param success: Optimization successful? (True/False)
        :param message: return message
        :return: OptimizeResult object
        """
        X, Y = self.acquisition.data

        # Filter on constraints
        valid = self.acquisition.feasible_data_index()

        if not np.any(valid):
            return OptimizeResult(success=False,
                                  message="No evaluations satisfied the constraints")

        valid_X = X[valid, :]
        valid_Y = Y[valid, :]
        valid_Yo = valid_Y[:, self.acquisition.objective_indices()]

        # Here is the place to plug in pareto front if valid_Y.shape[1] > 1
        # else
        idx = np.argmin(valid_Yo)

        return OptimizeResult(x=valid_X[idx, :],
                              success=success,
                              fun=valid_Yo[idx, :],
                              message=message)

    def optimize(self, objectivefx, n_iter=20):
        """
        Run Bayesian optimization for a number of iterations. Each iteration a new data point is selected by optimizing
        an acquisition function. This point is evaluated on the objective and black-box constraints. This retrieved
        information then updates the model
        :param objectivefx: (list of) expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """
        fxs = np.atleast_1d(objectivefx)
        return super(BayesianOptimizer, self).optimize(lambda x: self._evaluate_objectives(x, fxs), n_iter=n_iter)

    def _optimize(self, fx, n_iter):
        """
        Internal optimization function. Receives and ObjectiveWrapper. As exclude_gradient is set to true,
        the placeholder created by _evaluate_objectives will not be returned.
        :param fx: ObjectiveWrapper object wrapping expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """

        # Evaluate and add the initial design (if any)
        initial = self.get_initial()
        values = fx(initial)
        self._update_model_data(initial, values)

        # Remove initial design for additional calls to optimize to proceed optimization
        self.set_initial(EmptyDesign(self.domain).generate())

        def inverse_acquisition(x):
            return tuple(map(lambda r: -r, self.acquisition.evaluate_with_gradients(np.atleast_2d(x))))

        # Optimization loop
        for i in range(n_iter):
            result = self.optimizer.optimize(inverse_acquisition)
            self._update_model_data(result.x, fx(result.x))

        return self._create_bo_result(True, "OK")
