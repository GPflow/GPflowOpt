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
from scipy.optimize import OptimizeResult, minimize
from GPflow import settings
import contextlib
import sys
import os

from .design import RandomDesign
from .objective import ObjectiveWrapper


class Optimizer(object):
    """
    An optimization algorithm.

    Starts from an initial (set of) point(s) it performs an optimization over a domain.
    May be gradient-based or gradient-free.
    """

    def __init__(self, domain, exclude_gradient=False):
        super(Optimizer, self).__init__()
        self._domain = domain
        self._initial = domain.value
        self._wrapper_args = dict(exclude_gradient=exclude_gradient)

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, dom):
        self._domain = dom
        self.set_initial(dom.value)

    def optimize(self, objectivefx, **kwargs):
        """
        Optimize a given function f over a domain.

        The optimizer class supports interruption. If during the optimization ctrl+c is pressed, the last best point is
        returned.
        
        The actual optimization routine is implemented in _optimize, to be implemented in subclasses.

        :param objectivefx: callable, taking one argument: a 2D numpy array. The number of columns correspond to the 
        dimensionality of the input domain.
        :return: OptimizeResult reporting the results.
        """
        objective = ObjectiveWrapper(objectivefx, **self._wrapper_args)
        try:
            result = self._optimize(objective, **kwargs)
        except KeyboardInterrupt:
            result = OptimizeResult(x=objective._previous_x,
                                    success=False,
                                    message="Caught KeyboardInterrupt, returning last good value.")
        result.x = np.atleast_2d(result.x)
        result.nfev = objective.counter
        return result

    def get_initial(self):
        """
        Return the initial set of points.
        """
        return self._initial

    def set_initial(self, initial):
        """
        Set the initial set of points. The dimensionality should match the domain dimensionality, and all points should 
        be within the domain
        :param initial: initial points, should all be within the domain of the optimizer.
        """
        initial = np.atleast_2d(initial)
        assert (initial in self.domain)
        self._initial = initial

    def gradient_enabled(self):
        """
        Returns if the optimizer is a gradient-based algorithm or not.
        """
        return not self._wrapper_args['exclude_gradient']

    @contextlib.contextmanager
    def silent(self):
        """
        Context for performing actions on an optimizer (such as optimize) with all stdout discarded.
        Usage example:

        >>> opt = BayesianOptimizer(domain, acquisition, optimizer)
        >>> with opt.silent():
        >>>     # Run without printing anything
        >>>     opt.optimize(fx, n_iter=2)
        """
        save_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        yield
        sys.stdout = save_stdout


class CandidateOptimizer(Optimizer):
    """
    Optimization of an objective function by evaluating a set of pre-defined candidate points.

    Returns the point with minimal objective value.

    For compatibility with the StagedOptimizer, the candidate points are concatenated with
    the initial points and evaluated.
    """

    def __init__(self, domain, candidates, batch=False):
        """
        :param domain: Optimization domain.
        :param candidates: candidate points, should be within the optimization domain. 
        :param batch: bool, evaluate the objective function on all points at once or one by one?
        """
        super(CandidateOptimizer, self).__init__(domain, exclude_gradient=True)
        assert(candidates in domain)
        self.candidates = candidates
        self._batch_mode = batch

    @Optimizer.domain.setter
    def domain(self, dom):
        # Attempt to transform candidates
        t = self.domain >> dom
        self.candidates = t.forward(self.candidates)
        self._domain = dom
        self.set_initial(dom.value)

    def get_initial(self):
        return np.vstack((super(CandidateOptimizer, self).get_initial(), self.candidates))

    def _evaluate_one_by_one(self, objective, X):
        """
        Evaluates each row of X individually.
        """
        return np.vstack(map(lambda x: objective(x), X))

    def _optimize(self, objective):
        points = self.get_initial()
        evaluations = objective(points) if self._batch_mode else self._evaluate_one_by_one(objective, points)
        idx_best = np.argmin(evaluations, axis=0)

        return OptimizeResult(x=points[idx_best, :],
                              success=True,
                              fun=evaluations[idx_best, :],
                              nfev=points.shape[0],
                              message="OK")


class MCOptimizer(CandidateOptimizer):
    """
    Optimization of an objective function by evaluating a set of random points.

    Note: each call to optimize, a different set of random points is evaluated.
    """

    def __init__(self, domain, nsamples, batch=False):
        super(MCOptimizer, self).__init__(domain, np.empty((0, domain.size)), batch=batch)
        self._nsamples = nsamples

    def _optimize(self, objective):
        self.candidates = RandomDesign(self._nsamples, self.domain).generate()
        return super(MCOptimizer, self)._optimize(objective)


class SciPyOptimizer(Optimizer):
    """
    Wraps SciPy's minimize function.
    """

    def __init__(self, domain, method='L-BFGS-B', tol=None, maxiter=1000):
        super(SciPyOptimizer, self).__init__(domain)
        options = dict(disp=settings.verbosity.optimisation_verb,
                       maxiter=maxiter)
        self.config = dict(tol=tol,
                           method=method,
                           options=options)

    def _optimize(self, objective):
        """
        Calls scipy.optimize.minimize. 
        """
        objective1d = lambda X: tuple(map(lambda arr: arr.ravel(), objective(X)))
        result = minimize(fun=objective1d,
                          x0=self.get_initial(),
                          jac=self.gradient_enabled(),
                          bounds=list(zip(self.domain.lower, self.domain.upper)),
                          **self.config)
        return result


class StagedOptimizer(Optimizer):
    """
    An optimization pipeline of multiple optimizers called in succession.

    A list of optimizers can be specified (all on the same domain). The optimal
    solution of the an optimizer is used as an initial point for the next optimizer.
    """

    def __init__(self, optimizers):
        assert all(map(lambda opt: optimizers[0].domain == opt.domain, optimizers))
        no_gradient = any(map(lambda opt: not opt.gradient_enabled(), optimizers))
        super(StagedOptimizer, self).__init__(optimizers[0].domain, exclude_gradient=no_gradient)
        self.optimizers = optimizers

    def optimize(self, objectivefx):
        """
        The StagedOptimizer overwrites the default behaviour of optimize(). It passes the best point of the previous
        stage to the next stage. If the optimization is interrupted or fails, this process stops and the OptimizeResult 
        is returned.
        """

        self.optimizers[0].set_initial(self.get_initial())
        fun_evals = []
        for current, next in zip(self.optimizers[:-1], self.optimizers[1:]):
            result = current.optimize(objectivefx)
            fun_evals.append(result.nfev)
            if not result.success:
                result.message += " StagedOptimizer interrupted after {0}.".format(current.__class__.__name__)
                break
            next.set_initial(result.x)

        if result.success:
            result = self.optimizers[-1].optimize(objectivefx)
            fun_evals.append(result.nfev)
        result.nfev = sum(fun_evals)
        result.nstages = len(fun_evals)
        return result
