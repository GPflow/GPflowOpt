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

from gpflow.param import Parameterized, DataHolder, AutoFlow
from gpflow import settings
from scipy.spatial.distance import pdist, squareform
import numpy as np
import tensorflow as tf

np_int_type = np_float_type = np.int32 if settings.dtypes.int_type is tf.int32 else np.int64
float_type = settings.dtypes.float_type
stability = settings.numerics.jitter_level


class BoundedVolumes(Parameterized):

    @classmethod
    def empty(cls, dim, dtype):
        """
        Returns an empty bounded volume (hypercube).

        :param dim: dimension of the volume
        :param dtype: dtype of the coordinates
        :return: an empty :class:`.BoundedVolumes`
        """
        setup_arr = np.zeros((0, dim), dtype=dtype)
        return cls(setup_arr.copy(), setup_arr.copy())

    def __init__(self, lb, ub):
        """
        Construct bounded volumes.

        :param lb: the lowerbounds of the volumes
        :param ub: the upperbounds of the volumes
        """
        super(BoundedVolumes, self).__init__()
        assert np.all(lb.shape == ub.shape)
        self.lb = DataHolder(np.atleast_2d(lb), 'pass')
        self.ub = DataHolder(np.atleast_2d(ub), 'pass')

    def append(self, lb, ub):
        """
        Add new bounded volumes.

        :param lb: the lowerbounds of the volumes
        :param ub: the upperbounds of the volumes
        """
        self.lb = np.vstack((self.lb.value, lb))
        self.ub = np.vstack((self.ub.value, ub))

    def clear(self):
        """
        Clears all stored bounded volumes
        """
        dtype = self.lb.value.dtype
        outdim = self.lb.shape[1]
        self.lb = np.zeros((0, outdim), dtype=dtype)
        self.ub = np.zeros((0, outdim), dtype=dtype)

    def size(self):
        """
        :return: volume of each bounded volume
        """
        return np.prod(self.ub.value - self.lb.value, axis=1)


def non_dominated_sort(objectives):
    """
    Computes the non-dominated set for a set of data points

    :param objectives: data points
    :return: tuple of the non-dominated set and the degree of dominance,
        dominances gives the number of dominating points for each data point
    """
    extended = np.tile(objectives, (objectives.shape[0], 1, 1))
    dominance = np.sum(np.logical_and(np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
                                      np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)

    return objectives[dominance == 0], dominance


class Pareto(Parameterized):
    def __init__(self, Y, threshold=0):
        """
        Construct a Pareto set.

        Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
        The latter is needed for certain multiobjective acquisition functions.
        E.g., the :class:`~.acquisition.HVProbabilityOfImprovement`.

        :param Y: output data points, size N x R
        :param threshold: approximation threshold for the generic divide and conquer strategy
            (default 0: exact calculation)
        """
        super(Pareto, self).__init__()
        self.threshold = threshold
        self.Y = Y

        # Setup data structures
        self.bounds = BoundedVolumes.empty(Y.shape[1], np_int_type)
        self.front = DataHolder(np.zeros((0, Y.shape[1])), 'pass')

        # Initialize
        self.update()

    @staticmethod
    def _is_test_required(smaller):
        """
        Tests if a point augments or dominates the Pareto set.

        :param smaller: a boolean ndarray storing test point < Pareto front
        :return: True if the test point dominates or augments the Pareto front (boolean)
        """
        # if and only if the test point is at least in one dimension smaller for every point in the Pareto set
        idx_dom_augm = np.any(smaller, axis=1)
        is_dom_augm = np.all(idx_dom_augm)

        return is_dom_augm

    def _update_front(self):
        """
        Calculate the non-dominated set of points based on the latest data.

        The stored Pareto set is sorted on the first objective in ascending order.

        :return: boolean, whether the Pareto set has actually changed since the last iteration
        """
        current = self.front.value
        pf, _ = non_dominated_sort(self.Y)

        self.front = pf[pf[:, 0].argsort(), :]

        return not np.array_equal(current, self.front.value)

    def update(self, Y=None, generic_strategy=False):
        """
        Update with new output data.

        Computes the Pareto set and if it has changed recalculates the cell bounds covering the non-dominated region.
        For the latter, a direct algorithm is used for two objectives, otherwise a
        generic divide and conquer strategy is employed.

        :param Y: output data points
        :param generic_strategy: Force the generic divide and conquer strategy regardless of the number of objectives
            (default False)
        """
        self.Y = Y if Y is not None else self.Y

        # Find (new) set of non-dominated points
        changed = self._update_front()

        # Recompute cell bounds if required
        # Note: if the Pareto set is based on model predictions it will almost always change in between optimizations
        if changed:
            # Clear data container
            self.bounds.clear()
            if generic_strategy:
                self.divide_conquer_nd()
            else:
                self.bounds_2d() if self.Y.shape[1] == 2 else self.divide_conquer_nd()

    def divide_conquer_nd(self):
        """
        Divide and conquer strategy to compute the cells covering the non-dominated region.

        Generic version: works for an arbitrary number of objectives.
        """
        outdim = self.Y.shape[1]

        # The divide and conquer algorithm operates on a pseudo Pareto set
        # that is a mapping of the real Pareto set to discrete values
        pseudo_pf = np.argsort(self.front.value, axis=0) + 1  # +1 as index zero is reserved for the ideal point

        # Extend front with the ideal and anti-ideal point
        min_pf = np.min(self.front.value, axis=0) - 1
        max_pf = np.max(self.front.value, axis=0) + 1

        pf_ext = np.vstack((min_pf, self.front.value, max_pf))  # Needed for early stopping check (threshold)
        pf_ext_idx = np.vstack((np.zeros(outdim, dtype=np_int_type),
                                pseudo_pf,
                                np.ones(outdim, dtype=np_int_type) * self.front.shape[0] + 1))

        # Start with one cell covering the whole front
        dc = [(np.zeros(outdim, dtype=np_int_type),
               (int(pf_ext_idx.shape[0]) - 1) * np.ones(outdim, dtype=np_int_type))]
        
        total_size = np.prod(max_pf - min_pf)

        # Start divide and conquer until we processed all cells
        while dc:
            # Process test cell
            cell = dc.pop()

            arr = np.arange(outdim)
            lb  = pf_ext[pf_ext_idx[cell[0], arr], arr]
            ub  = pf_ext[pf_ext_idx[cell[1], arr], arr]

            # Acceptance test:
            if self._is_test_required((ub - stability) < self.front.value):
                # Cell is a valid integral bound: store
                self.bounds.append(pf_ext_idx[cell[0], np.arange(outdim)],
                                   pf_ext_idx[cell[1], np.arange(outdim)])
            # Reject test:
            elif self._is_test_required((lb + stability) < self.front.value):
                # Cell can not be discarded: calculate the size of the cell
                dc_dist = cell[1] - cell[0]
                hc = BoundedVolumes(pf_ext[pf_ext_idx[cell[0], np.arange(outdim)], np.arange(outdim)],
                                    pf_ext[pf_ext_idx[cell[1], np.arange(outdim)], np.arange(outdim)])

                # Only divide when it is not an unit cell and the volume is above the approx. threshold
                if np.any(dc_dist > 1) and np.all((hc.size()[0] / total_size) > self.threshold):
                    # Divide the test cell over its largest dimension
                    edge_size, idx = np.max(dc_dist), np.argmax(dc_dist)
                    edge_size1 = int(np.round(edge_size / 2.0))
                    edge_size2 = edge_size - edge_size1

                    # Store divided cells
                    ub = np.copy(cell[1])
                    ub[idx] -= edge_size1
                    dc.append((np.copy(cell[0]), ub))

                    lb = np.copy(cell[0])
                    lb[idx] += edge_size2
                    dc.append((lb, np.copy(cell[1])))
            # else: cell can be discarded

    def bounds_2d(self):
        """
        Computes the cells covering the non-dominated region for the specific case of only two objectives.

        Assumes the Pareto set has been sorted in ascending order on the first objective.
        This implies the second objective is sorted in descending order.
        """
        outdim = self.Y.shape[1]
        assert outdim == 2

        pf_idx = np.argsort(self.front.value, axis=0)
        pf_ext_idx = np.vstack((np.zeros(outdim, dtype=np_int_type),
                                pf_idx + 1,
                                np.ones(outdim, dtype=np_int_type) * self.front.shape[0] + 1))

        for i in range(pf_ext_idx[-1, 0]):
            self.bounds.append((i, 0),
                               (i+1, pf_ext_idx[-i-1, 1]))

    @AutoFlow((float_type, [None]))
    def hypervolume(self, reference):
        """
        Autoflow method to calculate the hypervolume indicator

        The hypervolume indicator is the volume of the dominated region.

        :param reference: reference point to use
            Should be equal or bigger than the anti-ideal point of the Pareto set
            For comparing results across runs the same reference point must be used
        :return: hypervolume indicator (the higher the better)
        """

        min_pf = tf.reduce_min(self.front, 0, keep_dims=True)
        R = tf.expand_dims(reference, 0)
        pseudo_pf = tf.concat((min_pf, self.front, R), 0)
        D = tf.shape(pseudo_pf)[1]
        N = tf.shape(self.bounds.ub)[0]

        idx = tf.tile(tf.expand_dims(tf.range(D), -1),[1, N])
        ub_idx = tf.reshape(tf.stack([tf.transpose(self.bounds.ub), idx], axis=2), [N * D, 2])
        lb_idx = tf.reshape(tf.stack([tf.transpose(self.bounds.lb), idx], axis=2), [N * D, 2])
        ub = tf.reshape(tf.gather_nd(pseudo_pf, ub_idx), [D, N])
        lb = tf.reshape(tf.gather_nd(pseudo_pf, lb_idx), [D, N])
        hv = tf.reduce_sum(tf.reduce_prod(ub - lb, 0))
        return tf.reduce_prod(R - min_pf) - hv
