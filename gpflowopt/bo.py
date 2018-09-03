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

from contextlib import contextmanager

import numpy as np
from scipy.optimize import OptimizeResult
import tensorflow as tf
from gpflow.gpr import GPR

from .acquisition import Acquisition, MCMCAcquistion
from .design import Design, EmptyDesign
from .objective import ObjectiveWrapper
from .optim import Optimizer, SciPyOptimizer
from .pareto import non_dominated_sort
from .models import ModelWrapper


def jitchol_callback(models):
    """
    Increase the likelihood in case of Cholesky failures.

    This is similar to the use of jitchol in GPy. Default callback for BayesianOptimizer.
    Only usable on GPR models, other types are ignored.
    """
    for m in np.atleast_1d(models):
        if isinstance(m, ModelWrapper):
            jitchol_callback(m.wrapped)  # pragma: no cover

        if not isinstance(m, GPR):
            continue

        s = m.get_free_state()
        eKdiag = np.mean(np.diag(m.kern.compute_K_symm(m.X.value)))
        for e in [0] + [10**ex for ex in range(-6,-1)]:
            try:
                m.likelihood.variance = m.likelihood.variance.value + e * eKdiag
                m.optimize(maxiter=5)
                break
            except tf.errors.InvalidArgumentError:  # pragma: no cover
                m.set_state(s)


class BayesianOptimizer(Optimizer):
    """
    A traditional Bayesian optimization framework implementation.
    
    Like other optimizers, this optimizer is constructed for optimization over a domain.
    Additionally, it is configured with a separate optimizer for the acquisition function.
    """

    def __init__(self, domain, acquisition, optimizer=None, initial=None, scaling=True, hyper_draws=None,
                 callback=jitchol_callback, verbose=False):
        """
        :param Domain domain: The optimization space.
        :param Acquisition acquisition: The acquisition function to optimize over the domain.
        :param Optimizer optimizer: (optional) optimization approach for the acquisition function.
            If not specified, :class:`~.optim.SciPyOptimizer` is used.
            This optimizer will run on the same domain as the :class:`.BayesianOptimizer` object.
        :param Design initial: (optional) The initial design of candidates to evaluate
            before the optimization loop runs. Note that if the underlying model contains already some data from
            an initial design, it is augmented with the evaluations obtained by evaluating
            the points as specified by the design.
        :param bool scaling: (boolean, default true) if set to true, the outputs are normalized, and the inputs are
            scaled to a unit cube. This only affects model training: calls to acquisition.data, as well as
            returned optima are unscaled (see :class:`~.DataScaler` for more details.). Note, the models contained by
            acquisition are modified directly, and so the references to the model outside of BayesianOptimizer now point
            to scaled models.
        :param int hyper_draws: (optional) Enable marginalization of model hyperparameters. By default, point estimates are
            used. If this parameter set to n, n hyperparameter draws from the likelihood distribution
            are obtained using Hamiltonian MC.
            (see `GPflow documentation <https://gpflow.readthedocs.io/en/latest//>`_ for details) for each model.
            The acquisition score is computed for each draw, and averaged.
        :param callable callback: (optional) this function or object will be called, after the
            data of all models has been updated with all models as retrieved by acquisition.models as argument without
            the wrapping model handling any scaling . This allows custom model optimization strategies to be implemented.
            All manipulations of GPflow models are permitted. Combined with the optimize_restarts parameter of
            :class:`~.Acquisition` this allows several scenarios: do the optimization manually from the callback
            (optimize_restarts equals 0), or choose the starting point + some random restarts (optimize_restarts > 0).
        """
        assert isinstance(acquisition, Acquisition)
        assert hyper_draws is None or hyper_draws > 0
        assert optimizer is None or isinstance(optimizer, Optimizer)
        assert initial is None or isinstance(initial, Design)
        super(BayesianOptimizer, self).__init__(domain, exclude_gradient=True)

        self._scaling = scaling
        if self._scaling:
            acquisition.enable_scaling(domain)

        self.acquisition = acquisition if hyper_draws is None else MCMCAcquistion(acquisition, hyper_draws)

        self.optimizer = optimizer or SciPyOptimizer(domain)
        self.optimizer.domain = domain
        initial = initial or EmptyDesign(domain)
        self.set_initial(initial.generate())

        self._model_callback = callback
        self.verbose = verbose

    @Optimizer.domain.setter
    def domain(self, dom):
        assert self.domain.size == dom.size
        super(BayesianOptimizer, self.__class__).domain.fset(self, dom)
        if self._scaling:
            self.acquisition.enable_scaling(dom)

    def _update_model_data(self, newX, newY):
        """
        Update the underlying models of the acquisition function with new data.

        :param newX: samples, size N x D
        :param newY: values obtained by evaluating the objective and constraint functions, size N x R
        """
        assert self.acquisition.data[0].shape[1] == newX.shape[-1]
        assert self.acquisition.data[1].shape[1] == newY.shape[-1]
        assert newX.shape[0] == newY.shape[0]
        if newX.size == 0:
            return
        X = np.vstack((self.acquisition.data[0], newX))
        Y = np.vstack((self.acquisition.data[1], newY))
        self.acquisition.set_data(X, Y)

    def _evaluate_objectives(self, X, fxs):
        """
        Evaluates a list of n functions on X.
        
        Returns a matrix, size N x sum(Q0,...Qn-1)
        with Qi the number of columns obtained by evaluating the i-th function.
       
        :param X: input points, size N x D
        :param fxs: functions, size n
        :return: tuple:
            (0) the evaluations Y, size N x sum(Q0,...Qn-1).
            (1) Not used, size N x 0. Bayesian Optimizer is gradient-free, however calling optimizer of the parent class
            expects a gradient. Will be discarded further on.
        """
        if X.size > 0:
            evaluations = np.hstack(map(lambda f: f(X), fxs))
            assert evaluations.shape[1] == self.acquisition.data[1].shape[1]
            return evaluations, np.zeros((X.shape[0], 0))
        else:
            return np.empty((0, self.acquisition.data[1].shape[1])), np.zeros((0, 0))

    def _create_bo_result(self, success, message):
        """
        Analyzes all data evaluated during the optimization, and return an `OptimizeResult`. Constraints are taken
        into account. The contents of x, fun, and constraints depend on the detected scenario:
        - single-objective: the best optimum of the feasible samples (if none, optimum of the infeasible samples)
        - multi-objective: the Pareto set of the feasible samples
        - only constraints: all the feasible samples (can be empty)

        In all cases, if not one sample satisfies all the constraints a message will be given and success=False.

        Do note that the feasibility check is based on the model predictions, but the constrained field contains
        actual data values.
       
        :param success: Optimization successful? (True/False)
        :param message: return message
        :return: OptimizeResult object
        """
        X, Y = self.acquisition.data

        # Filter on constraints
        valid = self.acquisition.feasible_data_index()

        # Extract the samples that satisfies all constraints
        if np.any(valid):
            X = X[valid, :]
            Y = Y[valid, :]
        else:
            success = False
            message = "No evaluations satisfied all the constraints"

        # Split between objectives and constraints
        Yo = Y[:, self.acquisition.objective_indices()]
        Yc = Y[:, self.acquisition.constraint_indices()]

        # Differentiate between different scenarios
        if Yo.shape[1] == 1:  # Single-objective: minimum
            idx = np.argmin(Yo)
        elif Yo.shape[1] > 1:  # Multi-objective: Pareto set
            _, dom = non_dominated_sort(Yo)
            idx = dom == 0
        else:  # Constraint satisfaction problem: all samples satisfying the constraints
            idx = np.arange(Yc.shape[0])

        return OptimizeResult(x=X[idx, :],
                              success=success,
                              fun=Yo[idx, :],
                              constraints=Yc[idx, :],
                              message=message)

    def optimize(self, objectivefx, n_iter=20):
        """
        Run Bayesian optimization for a number of iterations.
        
        Before the loop is initiated, first all points retrieved by :meth:`~.optim.Optimizer.get_initial` are evaluated
        on the objective and black-box constraints. These points are then added to the acquisition function 
        by calling :meth:`~.acquisition.Acquisition.set_data` (and hence, the underlying models). 
        
        Each iteration a new data point is selected for evaluation by optimizing an acquisition function. This point
        updates the models.
        
        :param objectivefx: (list of) expensive black-box objective and constraint functions. For evaluation, the 
            responses of all the expensive functions are aggregated column wise.
            Unlike the typical :class:`~.optim.Optimizer` interface, these functions should not return gradients. 
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """
        fxs = np.atleast_1d(objectivefx)
        return super(BayesianOptimizer, self).optimize(lambda x: self._evaluate_objectives(x, fxs), n_iter=n_iter)

    def _optimize(self, fx, n_iter):
        """
        Internal optimization function. Receives an ObjectiveWrapper as input. As exclude_gradient is set to true,
        the placeholder created by :meth:`_evaluate_objectives` will not be returned.
       
        :param fx: :class:`.objective.ObjectiveWrapper` object wrapping expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """
        assert isinstance(fx, ObjectiveWrapper)

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
            # If a callback is specified, and acquisition has the setup flag enabled (indicating an upcoming
            # compilation), run the callback.
            with self.silent():
                if self._model_callback and self.acquisition._needs_setup:
                    self._model_callback([m.wrapped for m in self.acquisition.models])

                result = self.optimizer.optimize(inverse_acquisition)
                self._update_model_data(result.x, fx(result.x))

            if self.verbose:
                metrics = []

                with self.silent():
                    bo_result = self._create_bo_result(True, 'Monitor')
                    metrics += ['MLL [' + ', '.join('{:.3}'.format(model.compute_log_likelihood()) for model in self.acquisition.models) + ']']

                # fmin
                n_points = bo_result.fun.shape[0]
                if n_points > 0:
                    funs = np.atleast_1d(np.min(bo_result.fun, axis=0))
                    fmin = 'fmin [' + ', '.join('{:.3}'.format(fun) for fun in funs) + ']'
                    if n_points > 1:
                        fmin += ' (size {0})'.format(n_points)

                    metrics += [fmin]

                # constraints
                n_points = bo_result.constraints.shape[0]
                if n_points > 0:
                    constraints = np.atleast_1d(np.min(bo_result.constraints, axis=0))
                    metrics += ['constraints [' + ', '.join('{:.3}'.format(constraint) for constraint in constraints) + ']']

                # error messages
                metrics += [r.message.decode('utf-8') if isinstance(r.message, bytes) else r.message for r in [bo_result, result] if not r.success]

                print('iter #{0:>3} - {1}'.format(
                    i,
                    ' - '.join(metrics)))

        return self._create_bo_result(True, "OK")

    @contextmanager
    def failsafe(self):
        """
        Context to provide a safe way for optimization.
        
        If a RuntimeError is generated, the data of the acquisition object is saved to the disk.
        in the current directory. This allows the data to be re-used (which makes sense for expensive data).

        The data can be used to experiment with fitting a GPflow model first (analyse the data, set sensible initial
        hyperparameter values and hyperpriors) before retrying Bayesian Optimization again.
        """
        try:
            yield
        except Exception as e:
            np.savez('failed_bopt_{0}'.format(id(e)), X=self.acquisition.data[0], Y=self.acquisition.data[1])
            raise
