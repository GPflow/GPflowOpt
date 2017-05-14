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
import GPflow

from .domain import ContinuousParameter


class Design(object):
    """
    Space-filling designs generated within a domain.
    """

    def __init__(self, size, domain):
        super(Design, self).__init__()
        self._size = size
        self.domain = domain

    @property
    def size(self):
        return self._size

    def generate(self):
        raise NotImplementedError


class RandomDesign(Design):
    """
    Random space-filling design
    """

    def __init__(self, size, domain):
        super(RandomDesign, self).__init__(size, domain)

    def generate(self):
        X = np.random.rand(self.size, self.domain.size)
        return X * (self.domain.upper - self.domain.lower) + self.domain.lower


class FactorialDesign(Design):
    """
    Grid-based design
    """

    def __init__(self, levels, domain):
        self.levels = levels
        size = levels ** domain.size
        super(FactorialDesign, self).__init__(size, domain)

    def generate(self):
        Xs = np.meshgrid(*[np.linspace(l, u, self.levels) for l, u in zip(self.domain.lower, self.domain.upper)])
        return np.vstack(map(lambda X: X.ravel(), Xs)).T


class EmptyDesign(Design):
    """
    No design, used as placeholder
    """

    def __init__(self, domain):
        super(EmptyDesign, self).__init__(0, domain)

    def generate(self):
        return np.empty((0, self.domain.size))


class DPPDesign(FactorialDesign):
    """
    Samples from a determinantal point process (DPP). This implementation represents the k-DPP exact sampling 
    methodology presented in
    ::
       @article{Kulesza:2012,
            title={Determinantal point processes for machine learning},
            author={Kulesza, Alex and Taskar, Ben and others},
            journal={Foundations and Trends in Machine Learning},
            volume={5},
            number={2--3},
            pages={123--286},
            year={2012},
            publisher={Now Publishers, Inc.}
       }
    """

    def __init__(self, size, domain):
        super(DPPDesign, self).__init__(size, domain)
        M = super(DPPDesign, self).generate()
        gauss_kern = GPflow.kernels.RBF(domain.size, ARD=False, lengthscales=[1])
        unit_cube = np.sum([ContinuousParameter('x{0}'.format(i), 0, 1) for i in np.arange(self.domain.size)])
        unit_scaling = domain >> unit_cube
        K = gauss_kern.compute_K_symm(unit_scaling.forward(M))

        self.L_decomp = tuple(map(lambda l: np.real(l), np.linalg.eig(K)))
        self._k = size

    @Design.size.getter
    def size(self):
        return self._k

    def _elementary_symmetric_polynomials(self):
        D = self.L_decomp[0]
        E = np.vstack((np.ones((1, D.size + 1)), np.hstack((np.zeros((self.size, 1)), np.diag(np.cumprod(D))[:self.size,:]))))
        for i in np.arange(1, self.size + 1):
            E[i, i:] = np.cumsum(np.hstack((E[i, i], D[i:] * E[i - 1, i:-1])))
        return E

    def _draw_k_mask(self):
        D = self.L_decomp[0]
        E = self._elementary_symmetric_polynomials()
        print(E.shape)
        r = self.size
        j = D.size
        mask = np.zeros((D.size,), dtype=bool)
        while r:
            T = 1 if j == r else D[j-1] * E[r-1, j-1] / E[r, j]
            j -= 1
            if np.random.rand(1) > T:
                continue
            mask[j] = True
            r -= 1

        return mask

    def generate(self):
        V = self.L_decomp[1]
        mask = self._draw_k_mask()

        V = V[:, mask]
        Y = np.zeros(np.sum(mask), dtype=np.int32)

        for i in np.arange(Y.size - 1, -1, -1):
            probs = np.sum(np.square(V), axis=1) / np.sum(np.square(V))
            Y[i] = np.where(np.random.rand(1) <= np.cumsum(probs))[0][0]
            j = np.where(V[Y[i], :])[0][0]

            Vc = np.tile(V[:, [j]], (1, V.shape[1] - 1)) * np.delete(V[Y[i], :], j) / V[Y[i], j]
            V = np.delete(V, j, axis=1) - Vc
            if V.size > 0:
                V = np.linalg.qr(V)[0]

        M = super(DPPDesign, self).generate()
        return M[np.sort(Y), :]
