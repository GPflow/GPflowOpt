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

from GPflow.param import Parameterized, AutoFlow, ParamList
from GPflow import settings

import numpy as np
import tensorflow as tf

import copy

float_type = settings.dtypes.float_type


class Acquisition(Parameterized):
    """
    An acquisition function maps the belief represented by a Bayesian model into a
    score indicating how promising a point is for evaluation.

    In Bayesian Optimization this function is typically optimized over the optimization domain
    to determine the next point for evaluation.

    An object of this class holds a list of GPflow models. For single objective optimization this is typically a 
    single model. Subclasses implement a build_acquisition function which computes the acquisition function (usually 
    from the predictive distribution) using TensorFlow. Each model is automatically optimized when an acquisition object
    is constructed or when set_data is called.

    Acquisition functions can be combined through addition or multiplication to construct joint criteria 
    (for instance for constrained optimization)
    """

    def __init__(self, models=[], optimize_restarts=5):
        super(Acquisition, self).__init__()
        self._models = ParamList([DataScaler(m) for m in np.atleast_1d(models).tolist()])
        self._default_params = list(map(lambda m: m.get_free_state(), self._models))

        assert (optimize_restarts >= 0)
        self.optimize_restarts = optimize_restarts
        self._optimize_models()

    def _optimize_models(self):
        """
        Optimizes the hyperparameters of all models that the acquisition function is based on.

        It is called after initialization and set_data(), and before optimizing the acquisition function itself.

        For each model the hyperparameters of the model at the time it was passed to __init__() are used as initial
        point and optimized. If optimize_restarts was configured to values larger than one additional randomization
        steps are performed.

        As a special case, if optimize_restarts is set to zero, the hyperparameters of the models are not optimized.
        This is useful when the hyperparameters are sampled using MCMC.
        """
        if self.optimize_restarts == 0:
            return

        for model, hypers in zip(self.models, self._default_params):
            runs = []
            for i in range(self.optimize_restarts):
                model.randomize() if i > 0 else model.set_state(hypers)
                try:
                    result = model.optimize()
                    runs.append(result)
                except tf.errors.InvalidArgumentError:  # pragma: no cover
                    print("Warning: optimization restart {0}/{1} failed".format(i + 1, self.optimize_restarts))
            if not runs:
                raise RuntimeError("All model hyperparameter optimization restarts failed, exiting.")
            best_idx = np.argmin([r.fun for r in runs])
            model.set_state(runs[best_idx].x)

    def build_acquisition(self):
        raise NotImplementedError

    def enable_scaling(self, domain):
        """
        Enables and configures the :class:`.DataScaler` objects wrapping the GP models.
        :param domain: :class:`.Domain` object, the input transform of the data scalers is configured as a transform
         from domain to the unit cube with the same dimensionality.
        """
        n_inputs = self.data[0].shape[1]
        assert (domain.size == n_inputs)
        for m in self.models:
            m.input_transform = domain >> UnitCube(n_inputs)
            m.normalize_output = True
        self._optimize_models()

    def set_data(self, X, Y):
        """
        Update the training data of the contained models. Automatically triggers a hyperparameter optimization
        step by calling _optimize_all() and an update of pre-computed quantities by calling setup().

        Consider Q to be the the sum of the output dimensions of the contained models, Y should have a minimum of
        Q columns. Only the first Q columns of Y are used while returning the scalar Q

        :param X: input data N x D
        :param Y: Responses N x M (M >= Q)
        :return: Q (sum of output dimensions of contained models)
        """
        num_outputs_sum = 0
        for model in self.models:
            num_outputs = model.Y.shape[1]
            Ypart = Y[:, num_outputs_sum:num_outputs_sum + num_outputs]
            num_outputs_sum += num_outputs

            model.X = X
            model.Y = Ypart

        self._optimize_models()
        if self.highest_parent == self:
            self.setup()
        return num_outputs_sum

    @property
    def models(self):
        return self._models

    @property
    def data(self):
        """
        Property for accessing the training data of the models.

        Corresponds to the input data X which is the same for every model,
        and column-wise concatenation of the Y data over all models

        :return: X, Y tensors (if in tf_mode) or X, Y numpy arrays.
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
        By default all outputs are objectives
        """
        return np.setdiff1d(np.arange(self.data[1].shape[1]), self.constraint_indices())

    def feasible_data_index(self):
        """
        Returns a boolean array indicating which data points are considered feasible (according to the acquisition
        function(s) ) and which not.
        By default all data is considered feasible
        :return: boolean ndarray, N
        """
        return np.ones(self.data[0].shape[0], dtype=bool)

    def setup(self):
        """
        Method triggered after calling set_data().

        Override for pre-calculation of quantities used later in
        the evaluation of the acquisition function for candidate points
        """
        pass

    @AutoFlow((float_type, [None, None]))
    def evaluate_with_gradients(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, also returns the gradients.
        """
        acq = self.build_acquisition(Xcand)
        return acq, tf.gradients(acq, [Xcand], name="acquisition_gradient")[0]

    @AutoFlow((float_type, [None, None]))
    def evaluate(self, Xcand):
        """
        AutoFlow method to compute the acquisition scores for candidates, without returning the gradients.
        """
        return self.build_acquisition(Xcand)

    def __add__(self, other):
        """
        Operator for adding acquisition functions. Example:

        >>> a1 = GPflowOpt.acquisition.ExpectedImprovement(m1)
        >>> a2 = GPflowOpt.acquisition.ProbabilityOfFeasibility(m2)
        >>> type(a1 + a2)
        <type 'GPflowOpt.acquisition.AcquisitionSum'>

        """
        if isinstance(other, AcquisitionSum):
            return AcquisitionSum([self] + other.operands.sorted_params)
        return AcquisitionSum([self, other])

    def __mul__(self, other):
        """
        Operator for multiplying acquisition functions. Example:

        >>> a1 = GPflowOpt.acquisition.ExpectedImprovement(m1)
        >>> a2 = GPflowOpt.acquisition.ProbabilityOfFeasibility(m2)
        >>> type(a1 * a2)
        <type 'GPflowOpt.acquisition.AcquisitionProduct'>

        """
        if isinstance(other, AcquisitionProduct):
            return AcquisitionProduct([self] + other.operands.sorted_params)
        return AcquisitionProduct([self, other])


class AcquisitionAggregation(Acquisition):
    """
    Special acquisition implementation for aggregating multiple others, using a TensorFlow reduce operation.
    """

    def __init__(self, operands, oper):
        """
        Constructor
        :param operands: list of acquisition objects
        :param oper: a tf.reduce operation (e.g., tf.reduce_sum) for aggregating the returned scores of each operand.
        """
        super(AcquisitionAggregation, self).__init__()
        assert (all([isinstance(x, Acquisition) for x in operands]))
        self.operands = ParamList(operands)
        self._oper = oper
        self.setup()

    def _optimize_models(self):
        pass

    @Acquisition.models.getter
    def models(self):
        return ParamList([model for acq in self.operands for model in acq.models.sorted_params])

    def enable_scaling(self, domain):
        for oper in self.operands:
            oper.enable_scaling(domain)

    def set_data(self, X, Y):
        offset = 0
        for operand in self.operands:
            offset += operand.set_data(X, Y[:, offset:])
        if self.highest_parent == self:
            self.setup()
        return offset

    def setup(self):
        for oper in self.operands:
            oper.setup()

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
    Acquisition object to apply MCMC over the hyperparameters of the models. The models of the acquisition object passed
    into an object of this class is optimized with MLE, and then sampled with HMC. These hyperparameter samples are then
    set in copies of the acquisition.

    To compute the acquisition, the predictions of the acquisition copies are averaged.
    """

    def __init__(self, acquisition, n_slices, **kwargs):
        assert isinstance(acquisition, Acquisition)
        assert n_slices > 0

        copies = [copy.deepcopy(acquisition) for _ in range(n_slices - 1)]
        for c in copies:
            c.optimize_restarts = 0

        # the call to the constructor of the parent classes, will optimize acquisition, so it obtains the MLE solution.
        super(MCMCAcquistion, self).__init__([acquisition] + copies)
        self._sample_opt = kwargs
        # Update the hyperparameters of the copies using HMC
        self._update_hyper_draws()

    def _update_hyper_draws(self):
        # Sample each model of the acquisition function - results in a list of 2D ndarrays.
        hypers = np.hstack([model.sample(len(self.operands), **self._sample_opt) for model in self.models])

        # Now visit all copies, and set state
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
        self._update_hyper_draws()
        if self.highest_parent == self:
            self.setup()
        return offset

    def build_acquisition(self, Xcand):
        # Average the predictions of the copies.
        return 1. / len(self.operands) * super(MCMCAcquistion, self).build_acquisition(Xcand)
