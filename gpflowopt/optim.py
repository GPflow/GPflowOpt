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

import contextlib
import os
import sys
import warnings

import numpy as np
from gpflow import settings
from scipy.optimize import OptimizeResult, minimize

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
        """
        The current domain the optimizer operates on.
        
        :return: :class:'~.domain.Domain` object 
        """
        return self._domain

    @domain.setter
    def domain(self, dom):
        """
        Sets a new domain for the optimizer.
        
        Resets the initial points to the middle of the domain.
        
        :param dom: new :class:'~.domain.Domain` 
        """
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
        
        :return: initial set of points, size N x D
        """
        return self._initial

    def set_initial(self, initial):
        """
        Set the initial set of points.
        
        The dimensionality should match the domain dimensionality, and all points should 
        be within the domain.
    
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


class MCOptimizer(Optimizer):
    """
    Optimization of an objective function by evaluating a set of random points.

    Note: each call to optimize, a different set of random points is evaluated.
    """

    def __init__(self, domain, nsamples):
        """
        :param domain: Optimization :class:`~.domain.Domain`.
        :param nsamples: number of random points to use
        """
        super(MCOptimizer, self).__init__(domain, exclude_gradient=True)
        self._nsamples = nsamples
        # Clear the initial data points
        self.set_initial(np.empty((0, self.domain.size)))

    @Optimizer.domain.setter
    def domain(self, dom):
        self._domain = dom

    def _get_eval_points(self):
        return RandomDesign(self._nsamples, self.domain).generate()

    def _optimize(self, objective):
        points = self._get_eval_points()
        evaluations = objective(points)
        idx_best = np.argmin(evaluations, axis=0)

        return OptimizeResult(x=points[idx_best, :],
                              success=True,
                              fun=evaluations[idx_best, :],
                              nfev=points.shape[0],
                              message="OK")

    def set_initial(self, initial):
        initial = np.atleast_2d(initial)
        if initial.size > 0:
            warnings.warn("Initial points set in {0} are ignored.".format(self.__class__.__name__), UserWarning)
            return

        super(MCOptimizer, self).set_initial(initial)


class CandidateOptimizer(MCOptimizer):
    """
    Optimization of an objective function by evaluating a set of pre-defined candidate points.

    Returns the point with minimal objective value.
    """

    def __init__(self, domain, candidates):
        """
        :param domain: Optimization :class:`~.domain.Domain`.
        :param candidates: candidate points, should be within the optimization domain.
        """
        super(CandidateOptimizer, self).__init__(domain, candidates.shape[0])
        assert (candidates in domain)
        self.candidates = candidates

    def _get_eval_points(self):
        return self.candidates

    @MCOptimizer.domain.setter
    def domain(self, dom):
        t = self.domain >> dom
        super(CandidateOptimizer, self.__class__).domain.fset(self, dom)
        self.candidates = t.forward(self.candidates)


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
        del self._initial

    @Optimizer.domain.setter
    def domain(self, domain):
        self._domain = domain
        for optimizer in self.optimizers:
            optimizer.domain = domain

    def _best_x(self, results):
        best_idx = np.argmin([r.fun for r in results if r.success])
        return results[best_idx].x, results[best_idx].fun

    def optimize(self, objectivefx):
        """
        The StagedOptimizer overwrites the default behaviour of optimize(). It passes the best point of the previous
        stage to the next stage. If the optimization is interrupted or fails, this process stops and the OptimizeResult 
        is returned.
        """

        results = []
        for current, following in zip(self.optimizers[:-1], self.optimizers[1:]):
            result = current.optimize(objectivefx)
            results.append(result)
            if not result.success:
                result.message += " StagedOptimizer interrupted after {0}.".format(current.__class__.__name__)
                break
            following.set_initial(self._best_x(results)[0])

        if result.success:
            result = self.optimizers[-1].optimize(objectivefx)
            results.append(result)

        result.nfev = sum(r.nfev for r in results)
        result.nstages = len(results)
        if any(r.success for r in results):
            result.x, result.fun = self._best_x(results)
        return result

    def get_initial(self):
        return self.optimizers[0].get_initial()

    def set_initial(self, initial):
        self.optimizers[0].set_initial(initial)
