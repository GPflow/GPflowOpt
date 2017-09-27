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

from ..scaling import DataScaler
from ..domain import UnitCube
from ..models import ModelWrapper

from gpflow.param import Parameterized, AutoFlow, ParamList
from gpflow.model import Model
from gpflow import settings

import numpy as np
import tensorflow as tf

import copy
from functools import wraps

float_type = settings.dtypes.float_type


def setup_required(method):
    """
    Decorator function to mark methods in Acquisition classes which require running setup if indicated by _needs_setup
    :param method: acquisition method
    """
    @wraps(method)
    def runnable(instance, *args, **kwargs):
        assert isinstance(instance, Acquisition)
        hp = instance.highest_parent
        if hp._needs_setup:
            # Avoid infinite loops, caused by setup() somehow invoking the evaluate on another acquisition
            # e.g. through feasible_data_index.
            hp._needs_setup = False

            # 1 - optimize
            hp._optimize_models()

            # 2 - setup
            hp._setup()
        results = method(instance, *args, **kwargs)
        return results

    return runnable


class Acquisition(Parameterized):
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

    def __init__(self, models=[], optimize_restarts=5):
        """
        :param models: list of GPflow models representing our beliefs about the problem
        :param optimize_restarts: number of optimization restarts to use when training the models
        """
        super(Acquisition, self).__init__()
        models = np.atleast_1d(models)
        assert all(isinstance(model, (Model, ModelWrapper)) for model in models)
        self._models = ParamList([DataScaler(m) for m in models])

        assert (optimize_restarts >= 0)
        self.optimize_restarts = optimize_restarts
        self._needs_setup = True

    def _optimize_models(self):
        """
        Optimizes the hyperparameters of all models that the acquisition function is based on.

        It is called automatically during initialization and each time :meth:`set_data` is called.
        When using the high-level :class:`..BayesianOptimizer` class calling :meth:`set_data` is taken care of.

        For each model the hyperparameters of the model at the time it was passed to __init__() are used as initial
        point and optimized. If optimize_restarts is set to >1, additional randomization
        steps are performed.

        As a special case, if optimize_restarts is set to zero, the hyperparameters of the models are not optimized.
        This is useful when the hyperparameters are sampled using MCMC.
        """
        if self.optimize_restarts == 0:
            return

        for model in self.models:
            runs = []
            for i in range(self.optimize_restarts):
                if i > 0:
                    model.randomize()
                try:
                    result = model.optimize()
                    runs.append(result)
                except tf.errors.InvalidArgumentError:  # pragma: no cover
                    print("Warning: optimization restart {0}/{1} failed".format(i + 1, self.optimize_restarts))
            if not runs:
                raise RuntimeError("All model hyperparameter optimization restarts failed, exiting.")
            best_idx = np.argmin([r.fun for r in runs])
            model.set_state(runs[best_idx].x)

    def build_acquisition(self, Xcand):
        raise NotImplementedError

    def enable_scaling(self, domain):
        """
        Enables and configures the :class:`.DataScaler` objects wrapping the GP models.
        
        Sets the _needs_setup attribute to True so the contained models are optimized and :meth:`setup` is run again
        right before evaluating the :class:`Acquisition` function. Note that the models are modified directly and
        references to them outside of the object will also point to scaled instances.

        :param domain: :class:`.Domain` object, the input transform of the data scalers is configured as a transform
            from domain to the unit cube with the same dimensionality.
        """
        n_inputs = self.data[0].shape[1]
        assert (domain.size == n_inputs)
        for m in self.models:
            m.input_transform = domain >> UnitCube(n_inputs)
            m.normalize_output = True
        self.highest_parent._needs_setup = True

    def set_data(self, X, Y):
        """
        Update the training data of the contained models

        Sets the _needs_setup attribute to True so the contained models are optimized and :meth:`setup` is run again
        right before evaluating the :class:`Acquisition` function.

        Let Q be the the sum of the output dimensions of all contained models, Y should have a minimum of
        Q columns. Only the first Q columns of Y are used while returning the scalar Q

        :param X: input data N x D
        :param Y: output data N x R (R >= Q)
        :return: Q (sum of output dimensions of contained models)
        """
        num_outputs_sum = 0
        for model in self.models:
            num_outputs = model.Y.shape[1]
            Ypart = Y[:, num_outputs_sum:num_outputs_sum + num_outputs]
            num_outputs_sum += num_outputs

            model.X = X
            model.Y = Ypart

        self.highest_parent._needs_setup = True
        return num_outputs_sum

    @property
    def models(self):
        """
        The GPflow models representing our beliefs of the optimization problem.
        
        :return: list of GPflow models 
        """
        return self._models.sorted_params

    @property
    def data(self):
        """
        The training data of the models.

        Corresponds to the input data X which is the same for every model,
        and column-wise concatenation of the Y data over all models

        :return: tuple X, Y of tensors (if in tf_mode) or numpy arrays.
        """
        if self._tf_mode:
            return self.models[0].X, tf.concat(list(map(lambda model: model.Y, self.models)), 1)
        else:
            return self.models[0].X.value, np.hstack(map(lambda model: model.Y.value, self.models))

    def constraint_indices(self):
        """
        Method returning the indices of the model outputs which correspond to the (expensive) constraint functions.
        By default there are no constraint functions
        """
        return np.empty((0,), dtype=int)

    def objective_indices(self):
        """
        Method returning the indices of the model outputs which are objective functions.
        
        By default all outputs are objectives.
        
        :return: indices to the objectives, size R
        """
        return np.setdiff1d(np.arange(self.data[1].shape[1]), self.constraint_indices())

    def feasible_data_index(self):
        """
        Returns a boolean array indicating which data points are considered feasible (according to the acquisition
        function(s) ) and which not.
        
        By default all data is considered feasible.
        
        :return: logical indices to the feasible data points, size N
        """
        return np.ones(self.data[0].shape[0], dtype=bool)

    def _setup(self):
        """
        Pre-calculation of quantities used later in the evaluation of the acquisition function for candidate points.
        
        Subclasses can implement this method to compute quantities (such as fmin). The decision when to run this function
        is governed by :class:`Acquisition`, based on the setup_required decorator on methods which require
        setup to be run (e.g. set_data).
        """
        pass

    def _setup_constraints(self):
        """
        Run only if some outputs handled by this acquisition are constraints. Used in aggregation.
        """
        if self.constraint_indices().size > 0:
            self._setup()

    def _setup_objectives(self):
        """
        Run only if all outputs handled by this acquisition are objectives. Used in aggregation.
        """
        if self.constraint_indices().size == 0:
            self._setup()

    @setup_required
    @AutoFlow((float_type, [None, None]))
    def evaluate_with_gradients(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, also returns the gradients.
        
        :return: acquisition scores, size N x 1
            the gradients of the acquisition scores, size N x D 
        """
        acq = self.build_acquisition(Xcand)
        return acq, tf.gradients(acq, [Xcand], name="acquisition_gradient")[0]

    @setup_required
    @AutoFlow((float_type, [None, None]))
    def evaluate(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, without returning the gradients.
        
        :return: acquisition scores, size N x 1
        """
        return self.build_acquisition(Xcand)

    def __add__(self, other):
        """
        Operator for adding acquisition functions. Example:

        >>> a1 = gpflowopt.acquisition.ExpectedImprovement(m1)
        >>> a2 = gpflowopt.acquisition.ProbabilityOfFeasibility(m2)
        >>> type(a1 + a2)
        <type 'gpflowopt.acquisition.AcquisitionSum'>
        """
        if isinstance(other, AcquisitionSum):
            return AcquisitionSum([self] + other.operands.sorted_params)
        return AcquisitionSum([self, other])

    def __mul__(self, other):
        """
        Operator for multiplying acquisition functions. Example:

        >>> a1 = gpflowopt.acquisition.ExpectedImprovement(m1)
        >>> a2 = gpflowopt.acquisition.ProbabilityOfFeasibility(m2)
        >>> type(a1 * a2)
        <type 'gpflowopt.acquisition.AcquisitionProduct'>
        """
        if isinstance(other, AcquisitionProduct):
            return AcquisitionProduct([self] + other.operands.sorted_params)
        return AcquisitionProduct([self, other])

    def __setattr__(self, key, value):
        super(Acquisition, self).__setattr__(key, value)
        if key is '_parent':
            self.highest_parent._needs_setup = True


class AcquisitionAggregation(Acquisition):
    """
    Aggregates multiple acquisition functions, using a TensorFlow reduce operation.
    """

    def __init__(self, operands, oper):
        """
        :param operands: list of acquisition objects
        :param oper: a tf.reduce operation (e.g., tf.reduce_sum) for aggregating the returned scores of each operand.
        """
        super(AcquisitionAggregation, self).__init__()
        assert (all([isinstance(x, Acquisition) for x in operands]))
        self.operands = ParamList(operands)
        self._oper = oper

    def _optimize_models(self):
        for oper in self.operands:
            oper._optimize_models()

    @Acquisition.models.getter
    def models(self):
        return [model for acq in self.operands for model in acq.models]

    def enable_scaling(self, domain):
        for oper in self.operands:
            oper.enable_scaling(domain)

    def set_data(self, X, Y):
        offset = 0
        for operand in self.operands:
            offset += operand.set_data(X, Y[:, offset:])
        return offset

    def _setup_constraints(self):
        for oper in self.operands:
            if oper.constraint_indices().size > 0:  # Small optimization, skip subtrees with objectives only
                oper._setup_constraints()

    def _setup_objectives(self):
        for oper in self.operands:
            oper._setup_objectives()

    def _setup(self):
        # Important: First setup acquisitions involving constraints
        self._setup_constraints()
        # Then objectives as these might depend on the constraint acquisition
        self._setup_objectives()

    def constraint_indices(self):
        offset = [0]
        idx = []
        for operand in self.operands:
            idx.append(operand.constraint_indices())
            offset.append(operand.data[1].shape[1])
        return np.hstack([i + o for i, o in zip(idx, offset[:-1])])

    def feasible_data_index(self):
        return np.all(np.vstack(map(lambda o: o.feasible_data_index(), self.operands)), axis=0)

    def build_acquisition(self, Xcand):
        return self._oper(tf.concat(list(map(lambda operand: operand.build_acquisition(Xcand), self.operands)), 1),
                          axis=1, keep_dims=True, name=self.__class__.__name__)

    def __getitem__(self, item):
        return self.operands[item]


class AcquisitionSum(AcquisitionAggregation):
    """
    Sum of acquisition functions
    """
    def __init__(self, operands):
        super(AcquisitionSum, self).__init__(operands, tf.reduce_sum)

    def __add__(self, other):
        if isinstance(other, AcquisitionSum):
            return AcquisitionSum(self.operands.sorted_params + other.operands.sorted_params)
        else:
            return AcquisitionSum(self.operands.sorted_params + [other])


class AcquisitionProduct(AcquisitionAggregation):
    """
    Product of acquisition functions
    """
    def __init__(self, operands):
        super(AcquisitionProduct, self).__init__(operands, tf.reduce_prod)

    def __mul__(self, other):
        if isinstance(other, AcquisitionProduct):
            return AcquisitionProduct(self.operands.sorted_params + other.operands.sorted_params)
        else:
            return AcquisitionProduct(self.operands.sorted_params + [other])


class MCMCAcquistion(AcquisitionSum):
    """
    Apply MCMC over the hyperparameters of an acquisition function (= over the hyperparameters of the contained models).
    
    The models passed into an object of this class are optimized with MLE (fast burn-in), and then further sampled with
    HMC. These hyperparameter samples are then set in copies of the acquisition.

    For evaluating the underlying acquisition function, the predictions of the acquisition copies are averaged.
    """

    def __init__(self, acquisition, n_slices, **kwargs):
        assert isinstance(acquisition, Acquisition)
        assert n_slices > 0
        # the call to the constructor of the parent classes, will optimize acquisition, so it obtains the MLE solution.
        super(MCMCAcquistion, self).__init__([acquisition]*n_slices)
        self._needs_new_copies = True
        self._sample_opt = kwargs

    def _optimize_models(self):
        # Optimize model #1
        self.operands[0]._optimize_models()

        # Copy it again if needed due to changed free state
        if self._needs_new_copies:
            new_copies = [copy.deepcopy(self.operands[0]) for _ in range(len(self.operands) - 1)]
            for c in new_copies:
                c.optimize_restarts = 0
            self.operands = ParamList([self.operands[0]] + new_copies)
            self._needs_new_copies = False

        # Draw samples using HMC
        # Sample each model of the acquisition function - results in a list of 2D ndarrays.
        hypers = np.hstack([model.sample(len(self.operands), **self._sample_opt) for model in self.models])

        # Now visit all acquisition copies, and set state
        for idx, draw in enumerate(self.operands):
            draw.set_state(hypers[idx, :])

    @Acquisition.models.getter
    def models(self):
        # Only return the models of the first operand, the copies remain hidden.
        return self.operands[0].models

    def set_data(self, X, Y):
        for operand in self.operands:
            # This triggers model.optimize() on self.operands[0]
            # All copies have optimization disabled, but must have update data.
            offset = operand.set_data(X, Y)
        return offset

    def build_acquisition(self, Xcand):
        # Average the predictions of the copies.
        return 1. / len(self.operands) * super(MCMCAcquistion, self).build_acquisition(Xcand)

    def _kill_autoflow(self):
        """
        Flag for recreation on next optimize.

        Following the recompilation of models, the free state might have changed. This means updating the samples can
        cause inconsistencies and errors.
        """
        super(MCMCAcquistion, self)._kill_autoflow()
        self._needs_new_copies = True
