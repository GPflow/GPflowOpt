# Copyright 2017 Joachim van der Herten, Ivo Couckuyt
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

from .acquisition import Acquisition
from ..pareto import Pareto

from gpflow.param import DataHolder
from gpflow import settings

import numpy as np
import tensorflow as tf

stability = settings.numerics.jitter_level
float_type = settings.dtypes.float_type


class HVProbabilityOfImprovement(Acquisition):
    """
    Hypervolume-based Probability of Improvement.

    A multiobjective acquisition function for multiobjective optimization. It is used to identify a complete Pareto set
    of non-dominated solutions.
    
    Key reference:

     ::

        @article{Couckuyt:2014,
            title={Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization},
            author={Couckuyt, Ivo and Deschrijver, Dirk and Dhaene, Tom},
            journal={Journal of Global Optimization},
            volume={60},
            number={3},
            pages={575--594},
            year={2014},
            publisher={Springer}
        }

    For a Pareto set :math:`\\mathcal{P}`, the non-dominated section of the objective space is denoted by :math:`A`.
    The :meth:`~..pareto.Pareto.hypervolume` of the dominated part of the space is denoted by :math:`\\mathcal{H}`
    and can be used as indicator for the optimality of the Pareto set (the higher the better).

    .. math::
       \\boldsymbol{\\mu} &= \\left[ \\mathbb{E} \\left[ f^{(1)}_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} \\right],
       ..., \\mathbb{E} \\left[ f^{(p)}_{\\star}\\,|\\, \\mathbf x, \\mathbf y, \\mathbf x_{\\star} \\right]\\right] \\\\
       I\\left(\\boldsymbol{\\mu}, \\mathcal{P}\\right) &=
       \\begin{cases} \\left( \\mathcal{H} \\left( \\mathcal{P} \\cup \\boldsymbol{\\mu} \\right) - \\mathcal{H}
       \\left( \\mathcal{P} \\right)) \\right) ~ if ~ \\boldsymbol{\\mu} \\in A
       \\\\ 0 ~ \\mbox{otherwise} \\end{cases} \\\\
       \\alpha(\\mathbf x_{\\star}) &= I\\left(\\boldsymbol{\\mu}, \\mathcal{P}\\right) p\\left(\\mathbf x_{\\star} \\in A \\right)

    Attributes:
        pareto: An instance of :class:`~..pareto.Pareto`.
    """

    def __init__(self, models):
        """
        :param models: A list of (possibly multioutput) GPflow representing our belief of the objectives.
        """
        super(HVProbabilityOfImprovement, self).__init__(models)
        num_objectives = self.data[1].shape[1]
        assert num_objectives > 1

        # Keep empty for now - it is updated in _setup()
        self.pareto = Pareto(np.empty((0, num_objectives)))
        self.reference = DataHolder(np.ones((1, num_objectives)))

    def _estimate_reference(self):
        pf = self.pareto.front.value
        f = np.max(pf, axis=0, keepdims=True) - np.min(pf, axis=0, keepdims=True)
        return np.max(pf, axis=0, keepdims=True) + 2 * f / pf.shape[0]

    def _setup(self):
        """
        Pre-computes the Pareto set and cell bounds for integrating over the non-dominated region.
        """
        super(HVProbabilityOfImprovement, self)._setup()

        # Obtain hypervolume cell bounds, use prediction mean
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        F = np.hstack((m.predict_f(feasible_samples)[0] for m in self.models))
        self.pareto.update(F)

        # Calculate reference point.
        self.reference = self._estimate_reference()

    def build_acquisition(self, Xcand):
        outdim = tf.shape(self.data[1])[1]
        num_cells = tf.shape(self.pareto.bounds.lb)[0]
        N = tf.shape(Xcand)[0]

        # Extended Pareto front
        pf_ext = tf.concat([-np.inf * tf.ones([1, outdim], dtype=float_type), self.pareto.front, self.reference], 0)

        # Predictions for candidates, concatenate columns
        preds = [m.build_predict(Xcand) for m in self.models]
        candidate_mean, candidate_var = (tf.concat(moment, 1) for moment in zip(*preds))
        candidate_var = tf.maximum(candidate_var, stability)  # avoid zeros

        # Calculate the cdf's for all candidates for every predictive distribution in the data points
        normal = tf.contrib.distributions.Normal(candidate_mean, tf.sqrt(candidate_var))
        Phi = tf.transpose(normal.cdf(tf.expand_dims(pf_ext, 1)), [1, 0, 2])  # N x pf_ext_size x outdim

        # tf.gather_nd indices for bound points
        col_idx = tf.tile(tf.range(outdim), (num_cells,))
        ub_idx = tf.stack((tf.reshape(self.pareto.bounds.ub, [-1]), col_idx), axis=1)  # (num_cells*outdim x 2)
        lb_idx = tf.stack((tf.reshape(self.pareto.bounds.lb, [-1]), col_idx), axis=1)  # (num_cells*outdim x 2)

        # Calculate PoI
        P1 = tf.transpose(tf.gather_nd(tf.transpose(Phi, perm=[1, 2, 0]), ub_idx))  # N x num_cell*outdim
        P2 = tf.transpose(tf.gather_nd(tf.transpose(Phi, perm=[1, 2, 0]), lb_idx))  # N x num_cell*outdim
        P = tf.reshape(P1 - P2, [N, num_cells, outdim])
        PoI = tf.reduce_sum(tf.reduce_prod(P, axis=2), axis=1, keep_dims=True)  # N x 1

        # Calculate Hypervolume contribution of points Y
        ub_points = tf.reshape(tf.gather_nd(pf_ext, ub_idx), [num_cells, outdim])
        lb_points = tf.reshape(tf.gather_nd(pf_ext, lb_idx), [num_cells, outdim])

        splus_valid = tf.reduce_all(tf.tile(tf.expand_dims(ub_points, 1), [1, N, 1]) > candidate_mean,
                                    axis=2)  # num_cells x N
        splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=float_type), -1)  # num_cells x N x 1
        splus_lb = tf.tile(tf.expand_dims(lb_points, 1), [1, N, 1])  # num_cells x N x outdim
        splus_lb = tf.maximum(splus_lb, candidate_mean)  # num_cells x N x outdim
        splus_ub = tf.tile(tf.expand_dims(ub_points, 1), [1, N, 1])  # num_cells x N x outdim
        splus = tf.concat([splus_idx, splus_ub - splus_lb], axis=2)  # num_cells x N x (outdim+1)
        Hv = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=2), axis=0, keep_dims=True))  # N x 1

        # return HvPoI
        return tf.multiply(Hv, PoI)
