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

from GPflow.param import Parameterized, AutoFlow, ParamList, DataHolder
from GPflow.model import Model
from GPflow import settings
from .scaling import DataScaler
from .domain import UnitCube

import numpy as np
import tensorflow as tf

import copy

float_type = settings.dtypes.float_type
stability = settings.numerics.jitter_level


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
        self._optimize_restarts = optimize_restarts
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
        if self._optimize_restarts == 0:
            return

        for model, hypers in zip(self.models, self._default_params):
            runs = []
            for i in range(self._optimize_restarts):
                model.randomize() if i > 0 else model.set_state(hypers)
                try:
                    result = model.optimize()
                    runs.append(result)
                except tf.errors.InvalidArgumentError:  # pragma: no cover
                    print("Warning: optimization restart {0}/{1} failed".format(i + 1, self._optimize_restarts))
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


class ExpectedImprovement(Acquisition):
    """
    Expected Improvement acquisition function for single-objective global optimization. 
    Introduced by (Mockus et al, 1975).

    Key reference:
    
    ::
    
       @article{Jones:1998,
            title={Efficient global optimization of expensive black-box functions},
            author={Jones, Donald R and Schonlau, Matthias and Welch, William J},
            journal={Journal of Global optimization},
            volume={13},
            number={4},
            pages={455--492},
            year={1998},
            publisher={Springer}
       }

    This acquisition function is the expectation of the improvement over the current best observation
    w.r.t. the predictive distribution. The definition is closely related to the Probability of Improvement,
    but adds a multiplication with the improvement w.r.t the current best observation to the integral.

    .. math::
       \\alpha(\\mathbf x_{\\star}) = \\int \\max(f_{\\min} - f_{\\star}, 0) \\, p( f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} ) \\, d f_{\\star}
    """

    def __init__(self, model):
        super(ExpectedImprovement, self).__init__(model)
        assert (isinstance(model, Model))
        self.fmin = DataHolder(np.zeros(1))
        self.setup()

    def setup(self):
        super(ExpectedImprovement, self).setup()
        # Obtain the lowest posterior mean for the previous - feasible - evaluations
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        samples_mean, _ = self.models[0].predict_f(feasible_samples)
        self.fmin.set_data(np.min(samples_mean, axis=0))

    def build_acquisition(self, Xcand):
        # Obtain predictive distributions for candidates
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, stability)

        # Compute EI
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        t1 = (self.fmin - candidate_mean) * normal.cdf(self.fmin)
        t2 = candidate_var * normal.prob(self.fmin)
        return tf.add(t1, t2, name=self.__class__.__name__)


class ProbabilityOfFeasibility(Acquisition):
    """
    Probability of Feasibility acquisition function for sampling feasible regions. Standard acquisition function for
    Bayesian Optimization with black-box expensive constraints. 

    Key reference:
    
    ::
    
       @article{parr2012infill,
            title={Infill sampling criteria for surrogate-based optimization with constraint handling},
            author={Parr, JM and Keane, AJ and Forrester, Alexander IJ and Holden, CME},
            journal={Engineering Optimization},
            volume={44},
            number={10},
            pages={1147--1166},
            year={2012},
            publisher={Taylor & Francis}
       }
    
    The acquisition function measures the probability of the latent function being smaller than 0 for a candidate point.
    
    .. math::
       \\alpha(\\mathbf x_{\\star}) = \\int_{-\\infty}^{0} \\, p(f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} ) \\, d f_{\\star}
    """

    def __init__(self, model, threshold=0.0, minimum_pof=0.5):
        """

        :param model: GPflow model (single output) for computing the PoF
        :param threshold: threshold value. Observed values lower than this value are considered valid
        :param minimum_pof: minimum pof score required for a point to be valid. For more information, see docstring
        of feasible_data_index
        """
        super(ProbabilityOfFeasibility, self).__init__(model)
        self.threshold = threshold
        self.minimum_pof = minimum_pof

    def constraint_indices(self):
        return np.arange(self.data[1].shape[1])

    def feasible_data_index(self):
        """
        Returns a boolean array indicating which points are feasible (True) and which are not (False)
        Answering the question *which points are feasible?* is slightly troublesome in case noise is present.
        Directly relying on the noisy data and comparing it to self.threshold does not make much sense.

        Instead, we rely on the model belief. More specifically, we evaluate the PoF (score between 0 and 1).
        As the implementation of the PoF corresponds to the cdf of the (normal) predictive distribution in
        a point evaluated at the threshold, requiring a minimum pof of 0.5 implies the mean of the predictive
        distribution is below the threshold, hence it is marked as feasible. A minimum pof of 0 marks all points valid.
        Setting it to 1 results in all invalid.
        :return: boolean ndarray, size N
        """
        # In
        pred = self.evaluate(self.data[0])
        return pred.ravel() > self.minimum_pof

    def build_acquisition(self, Xcand):
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, stability)
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        return normal.cdf(tf.constant(self.threshold, dtype=float_type), name=self.__class__.__name__)


class ProbabilityOfImprovement(Acquisition):
    """
    Probability of Improvement acquisition function for single-objective global optimization.

    .. math::
       \\alpha(\\mathbf x_{\\star}) = \\int_{-\\infty}^{f_{\\min}} \\, p( f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} ) \\, d f_{\\star}
    """

    def __init__(self, model):
        super(ProbabilityOfImprovement, self).__init__(model)
        self.fmin = DataHolder(np.zeros(1))
        self.setup()

    def setup(self):
        super(ProbabilityOfImprovement, self).setup()
        samples_mean, _ = self.models[0].predict_f(self.data[0])
        self.fmin.set_data(np.min(samples_mean, axis=0))

    def build_acquisition(self, Xcand):
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, stability)
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        return normal.cdf(self.fmin, name=self.__class__.__name__)


class LowerConfidenceBound(Acquisition):
    """
    Lower confidence bound acquisition function for single-objective global optimization.

    .. math::
       \\alpha(\\mathbf x_{\\star}) =\\mathbb{E} \\left[ f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} \\right]
       - \\sigma \\mbox{Var} \\left[ f_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} \\right]
    """

    def __init__(self, model, sigma=2.0):
        super(LowerConfidenceBound, self).__init__(model)
        self.sigma = sigma

    def build_acquisition(self, Xcand):
        candidate_mean, candidate_var = self.models[0].build_predict(Xcand)
        candidate_var = tf.maximum(candidate_var, 0)
        return tf.subtract(candidate_mean, self.sigma * tf.sqrt(candidate_var), name=self.__class__.__name__)


class AcquisitionAggregation(Acquisition):
    def __init__(self, operands, oper):
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
    def __init__(self, acquisition, n_slices, **kwargs):
        assert isinstance(acquisition, Acquisition)
        assert n_slices > 0

        copies = [copy.deepcopy(acquisition) for _ in range(n_slices - 1)]
        for c in copies:
            c._optimize_restarts = 0

        super(MCMCAcquistion, self).__init__([acquisition] + copies)
        self._sample_opt = kwargs
        self._update_hyper_draws()

    def _update_hyper_draws(self):
        # Sample each model of the acquisition function - results in a list of 2D ndarrays.
        hypers = np.hstack([model.sample(len(self.operands), **self._sample_opt) for model in self.models])

        # Now visit all copies, and set state
        for draw, idx in zip(self.operands, range(0, len(self.operands))):
            draw.set_state(hypers[idx, :])

    @Acquisition.models.getter
    def models(self):
        return self.operands[0].models

    def set_data(self, X, Y):
        for operand in self.operands:
            # this triggers model.optimimze() on self.operands[0]
            # All copies have optimization disabled.
            offset = operand.set_data(X, Y)
        self._update_hyper_draws()
        if self.highest_parent == self:
            self.setup()
        return offset

    def build_acquisition(self, Xcand):
        return 1. / len(self.operands) * super(MCMCAcquistion, self).build_acquisition(Xcand)
