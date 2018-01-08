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

from ..core import ICriterion
from ..decors import setup_required
from ..domain import UnitCube
from ..misc import randomize_model
from ..params import ModelWrapper
from ..scaling import DataScaler

from gpflow import Parameterized, autoflow, ParamList, settings, params_as_tensors
from gpflow.core import TensorConverter
from gpflow.models import Model
from gpflow.training import AdamOptimizer

import numpy as np
import tensorflow as tf

import abc
import copy


class Acquisition(Parameterized, ICriterion):
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

    def __init__(self, models=[], model_optimizer=None, optimize_restarts=5, maximize=True):
        """
        :param models: list of GPflow models representing our beliefs about the problem
        :param optimize_restarts: number of optimization restarts to use when training the models
        """
        super(Acquisition, self).__init__()
        models = np.atleast_1d(models)
        assert (optimize_restarts >= 0)
        assert all(isinstance(model, (Model, ModelWrapper)) for model in models)

        self.optimize_restarts = optimize_restarts
        if len(models) > 0:
            self.models = ParamList(self._wrap_models(models))
        #self._model_optimizer = model_optimizer or ScipyOptimizer()
        self._model_optimizer = AdamOptimizer(0.01)
        self._needs_setup = True
        self.maximize = maximize

    def _wrap_models(self, models):
        return [DataScaler(m) for m in models]

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

        for model in map(ModelWrapper.unwrap, self.models):
            runs = []
            for i in range(self.optimize_restarts):
                if i > 0:
                    randomize_model(model)
                try:
                    self._model_optimizer.minimize(model)
                    run_info = dict(score=model.compute_log_prior() + model.compute_log_likelihood(),
                                    state=copy.deepcopy(model.read_trainables()))
                    runs.append(run_info)
                except tf.errors.InvalidArgumentError:  # pragma: no cover
                    print("Warning: optimization restart {0}/{1} failed".format(1, self.optimize_restarts))

            best_idx = np.argmax([r['score'] for r in runs])
            model.assign(runs[best_idx]['state'])

    @abc.abstractmethod
    @params_as_tensors
    def _build_acquisition(self, Xcand):
        pass

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
            m.set_input_transform(domain >> UnitCube(n_inputs))
            m.normalize_output = True
        self.root._needs_setup = True

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

        self.root._needs_setup = True
        return num_outputs_sum

    @property
    def data(self):
        """
        The training data of the models.

        Corresponds to the input data X which is the same for every model,
        and column-wise concatenation of the Y data over all models

        :return: tuple X, Y of tensors (if in tf_mode) or numpy arrays.
        """
        if TensorConverter.tensor_mode(self):
            return self.models[0].X, tf.concat(list(map(lambda model: model.Y, self.models)), 1)
        else:
            X = self.models[0].X.read_value()
            Y = np.hstack(map(lambda model: model.Y.read_value(), self.models))
            return X, Y

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
    @autoflow((settings.tf_float, [None, None]))
    def evaluate_with_gradients(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, also returns the gradients.
        
        :return: acquisition scores, size N x 1
            the gradients of the acquisition scores, size N x D 
        """
        acq = self._build_acquisition(Xcand)
        return acq, tf.gradients(acq, [Xcand], name="acquisition_gradient")[0]

    @setup_required
    @autoflow((settings.tf_float, [None, None]))
    def evaluate(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, without returning the gradients.
        
        :return: acquisition scores, size N x 1
        """
        return self._build_acquisition(Xcand)

    def __add__(self, other):
        """
        Operator for adding acquisition functions. Example:

        >>> a1 = gpflowopt.acquisition.ExpectedImprovement(m1)
        >>> a2 = gpflowopt.acquisition.ProbabilityOfFeasibility(m2)
        >>> type(a1 + a2)
        <type 'gpflowopt.acquisition.AcquisitionSum'>
        """
        if isinstance(other, AcquisitionSum):
            return AcquisitionSum([self] + list(other.operands.params))
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
            return AcquisitionProduct([self] + list(other.operands.params))
        return AcquisitionProduct([self, other])

    def __setattr__(self, key, value):
        super(Acquisition, self).__setattr__(key, value)
        if key is '_parent':
            self.root._needs_setup = True


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

    @property
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

    @params_as_tensors
    def _build_acquisition(self, Xcand):
        return self._oper(tf.concat(list(map(lambda operand: operand._build_acquisition(Xcand), self.operands)), 1),
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
            return AcquisitionSum(list(self.operands.params) + list(other.operands.params))
        else:
            return AcquisitionSum(list(self.operands.params) + [other])


class AcquisitionProduct(AcquisitionAggregation):
    """
    Product of acquisition functions
    """
    def __init__(self, operands):
        super(AcquisitionProduct, self).__init__(operands, tf.reduce_prod)

    def __mul__(self, other):
        if isinstance(other, AcquisitionProduct):
            return AcquisitionProduct(list(self.operands.params) + list(other.operands.params))
        else:
            return AcquisitionProduct(list(self.operands.params) + [other])

