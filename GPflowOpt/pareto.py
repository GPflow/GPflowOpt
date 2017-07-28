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

from GPflow.param import Parameterized, DataHolder, AutoFlow
from GPflow import settings
from scipy.spatial.distance import pdist, squareform
import numpy as np
import tensorflow as tf

np_int_type = np_float_type = np.int32 if settings.dtypes.int_type is tf.int32 else np.int64
float_type = settings.dtypes.float_type


class BoundedVolumes(Parameterized):

    @classmethod
    def empty(cls, dim, dtype):
        setup_arr = np.zeros((0, dim), dtype=dtype)
        return cls(setup_arr.copy(), setup_arr.copy())

    def __init__(self, lb, ub):
        super(BoundedVolumes, self).__init__()
        assert np.all(lb.shape == lb.shape)
        self.lb = DataHolder(np.atleast_2d(lb), 'pass')
        self.ub = DataHolder(np.atleast_2d(ub), 'pass')

    def append(self, lb, ub):
        self.lb = np.vstack((self.lb.value, lb))
        self.ub = np.vstack((self.ub.value, ub))

    def clear(self):
        dtype = self.lb.value.dtype
        outdim = self.lb.shape[1]
        self.lb = np.zeros((0, outdim), dtype=dtype)
        self.ub = np.zeros((0, outdim), dtype=dtype)

    def size(self):
        return np.prod(self.ub.value - self.lb.value, axis=1)


def non_dominated_sort(objectives):
    # Dominance
    extended = np.tile(objectives, (objectives.shape[0], 1, 1))
    dominance = np.sum(np.logical_and(np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
                                      np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)

    return objectives[dominance == 0], dominance


class Pareto(Parameterized):
    def __init__(self, Y, threshold=0):
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
        # Test if test point augments or dominates the Pareto set
        # <=> test point is at least in one dimension smaller for every point in the Pareto set
        idx_dom_augm = np.any(smaller, axis=1)
        is_dom_augm = np.all(idx_dom_augm)

        return is_dom_augm

    def _update_front(self):
        """
        Calculate non-dominated set of points based on the latest data
        :return: whether the Pareto set has actually changed since the last iteration
        """
        current = self.front.value
        pf, _ = non_dominated_sort(self.Y)

        self.front = pf[pf[:, 0].argsort(), :]

        return not np.array_equal(current, self.front.value)

    def update(self, Y=None, generic_strategy=False):
        self.Y = Y if Y is not None else self.Y

        # Find (new) set of non-dominated points
        changed = self._update_front()

        # Recompute cell bounds if required
        # Note: if the Pareto set is based on model predictions it will almost always change in between optimizations
        if changed:
            # Clear data containers
            self.bounds.clear()
            if generic_strategy:
                self.divide_conquer()
            else:
                self.pareto2d_bounds() if self.Y.shape[1] == 2 else self.divide_conquer()

    def divide_conquer(self):
        """
        Divide and conquer strategy to compute the cells covering the non-dominated region

        Generic version, works for an arbitrary number of objectives
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
            cell = dc.pop(0)

            # Acceptance test:
            if self._is_test_required((cell[1] - 0.5) < pseudo_pf):
                # Cell is a valid integral bound: store
                self.bounds.append(pf_ext_idx[cell[0], np.arange(outdim)],
                                   pf_ext_idx[cell[1], np.arange(outdim)])
            # Reject test:
            elif self._is_test_required((cell[0] + 0.5) < pseudo_pf):
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

    def pareto2d_bounds(self):
        """
        Computes the cell bounds covering the non-dominated region

        for the specific case of only two objectives
        """
        outdim = self.Y.shape[1]

        pf_idx = np.argsort(self.front.value, axis=0)
        pf_ext_idx = np.vstack((np.zeros(outdim, dtype=np_int_type),
                                pf_idx + 1,
                                np.ones(outdim, dtype=np_int_type) * self.front.shape[0] + 1))

        for i in range(pf_ext_idx[-1, 0]):
            self.bounds.append((i, 0),
                               (i+1, pf_ext_idx[-i-1, 1]))

    @AutoFlow((float_type, [None]))
    def hypervolume(self, reference):

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
