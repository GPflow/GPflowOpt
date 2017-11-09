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
from scipy.spatial.distance import cdist, pdist
import tensorflow as tf

from gpflow import settings

from .domain import ContinuousParameter


float_type = settings.dtypes.float_type
stability = settings.numerics.jitter_level
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Design(object):
    """
    Design of size N (number of points) generated within a D-dimensional domain.
    
    Users should call generate() which auto-scales the design to the domain specified in the constructor.
    To implement new design methodologies subclasses should implement create_design(),
    which returns the design on the domain specified by the generative_domain method (which defaults to a unit cube).
    """

    def __init__(self, size, domain):
        """
        :param size: number of data points to generate
        :param domain: domain to generate data points in.
        """
        super(Design, self).__init__()
        self.size = size
        self.domain = domain

    @property
    def generative_domain(self):
        """
        :return: Domain object representing the domain associated with the points generated in create_design().
            Defaults to [0,1]^D, can be overwritten by subclasses
        """
        return np.sum([ContinuousParameter('d{0}'.format(i), 0, 1) for i in np.arange(self.domain.size)])

    def generate(self):
        """
        Creates the design in the domain specified during construction.

        It is guaranteed that all data points satisfy this domain

        :return: data matrix, size N x D
        """
        Xs = self.create_design()
        assert (Xs in self.generative_domain)
        assert (Xs.shape == (self.size, self.domain.size))
        transform = self.generative_domain >> self.domain
        # X = np.clip(transform.forward(Xs), self.domain.lower, self.domain.upper)
        X = transform.forward(Xs)
        assert (X in self.domain)
        return X

    def create_design(self):
        """
        Returns a design generated in the `generative` domain.

        This method should be implemented in the subclasses.
        
        :return: data matrix, N x D
        """
        raise NotImplementedError


class RandomDesign(Design):
    """
    Random space-filling design.

    Generates points drawn from the standard uniform distribution U(0,1).
    """

    def __init__(self, size, domain):
        super(RandomDesign, self).__init__(size, domain)

    def create_design(self):
        return np.random.rand(self.size, self.domain.size).astype(np_float_type)


class FactorialDesign(Design):
    """
    A k-level grid-based design.

    Design with the optimal minimal distance between points (a simple grid), however it risks collapsing points when
    removing parameters. Its size is a power of the domain dimensionality.
    """

    def __init__(self, levels, domain):
        self.levels = levels
        size = levels ** domain.size
        super(FactorialDesign, self).__init__(size, domain)

    @Design.generative_domain.getter
    def generative_domain(self):
        return self.domain

    def create_design(self):
        Xs = np.meshgrid(*[np.linspace(l, u, self.levels) for l, u in zip(self.domain.lower, self.domain.upper)])
        return np.vstack(map(lambda X: X.ravel(), Xs)).T


class EmptyDesign(Design):
    """
    No design, can be used as placeholder.
    """

    def __init__(self, domain):
        super(EmptyDesign, self).__init__(0, domain)

    def create_design(self):
        return np.empty((0, self.domain.size))


class LatinHyperCube(Design):
    """
    Latin hypercube with optimized minimal distance between points.

    Created with the Translational Propagation algorithm to avoid lengthy generation procedures.
    For dimensions smaller or equal to 6, this algorithm finds the quasi-optimal LHD with overwhelming probability.
    To increase this probability, if a design for a domain with dimensionality D is requested,
    D different designs are generated using seed sizes 1,2,...D (unless a maximum seed size 1<= S <= D is specified.
    The seeds themselves are small Latin hypercubes generated with the same algorithm.
    
    Beyond 6D, the probability of finding the optimal LHD fades, although the resulting designs are still acceptable. 
    Somewhere beyond 15D this algorithm tends to slow down a lot and become very memory demanding. Key reference is
    
    ::
    
       @article{Viana:2010,
            title={An algorithm for fast optimal Latin hypercube design of experiments},
            author={Viana, Felipe AC and Venter, Gerhard and Balabanov, Vladimir},
            journal={International Journal for Numerical Methods in Engineering},
            volume={82},
            number={2},
            pages={135--156},
            year={2010},
            publisher={John Wiley & Sons, Ltd.}
       }

    For pre-generated LHDs please see the `following website  <https://spacefillingdesigns.nl/>`_.
    """

    def __init__(self, size, domain, max_seed_size=None):
        """
        :param size: requested size N for the LHD 
        :param domain: domain to generate the LHD for, must be continuous
        :param max_seed_size: the maximum size 1 <= S <= D for the seed, . If unspecified, equals the dimensionality D 
            of the domain. During generation, S different designs are generated. Seeds with sizes 1,2,...S are used.
            Each seed itself is a small LHD.
        """
        super(LatinHyperCube, self).__init__(size, domain)
        self._max_seed_size = np.round(max_seed_size or domain.size)
        assert (1 <= np.round(self._max_seed_size) <= domain.size)

    @Design.generative_domain.getter
    def generative_domain(self):
        """
        :return: Domain object representing [1, N]^D, the generative domain for the TPLHD algorithm. 
        """
        return np.sum([ContinuousParameter('d{0}'.format(i), 1, self.size) for i in np.arange(self.domain.size)])

    def create_design(self):
        """
        Generate several LHDs with increasing seed.
        
        Maximum S = min(dimensionality,max_seed_size).
        From S candidate designs, the one with the best intersite distance is returned

        :return: data matrix, size N x D.
        """
        candidates = []
        scores = []

        for i in np.arange(1, min(self.size, self._max_seed_size) + 1):
            if i < 3:
                # Hardcoded seeds for 1 or two points.
                seed = np.arange(1, i + 1)[:, None] * np.ones((1, self.domain.size))
            else:
                # Generate larger seeds recursively by creating small TPLHD's
                seed = LatinHyperCube(i, self.domain, max_seed_size=i - 1).generate()

            # Create all designs and compute score
            X = self._tplhd_design(seed)
            candidates.append(X)
            scores.append(np.min(pdist(X)))

        # Transform best design (highest score) to specified domain
        return candidates[np.argmax(scores)]

    def _tplhd_design(self, seed):
        """
        Creates an LHD with the Translational propagation algorithm.
         
        Uses the specified seed and design size N specified during construction.

        :param seed: seed design, size S x D
        :return: data matrix, size N x D
        """
        ns, nv = seed.shape

        # Start by computing two quantities.
        # 1) the number of translation steps in each dimension
        nd = np.power(self.size / float(ns), 1 / float(nv))
        ndStar = np.ceil(nd)

        # 2) the total amount of points we'll be generating.
        # Typically, npStar > self.size, although sometimes npStar == self.size
        npStar = np.power(ndStar, nv) * ns if ndStar > nd else self.size

        # First rescale the seed, then perform translations and propagations.
        seed = self._rescale_seed(seed, npStar, ndStar)
        X = self._translate_propagate(seed, npStar, ndStar)

        # In case npStar > N, shrink the design to the requested size specified in __init__
        return self._shrink(X, self.size)

    @staticmethod
    def _rescale_seed(seed, npStar, ndStar):
        """
        Rescales the seed design

        :param seed: seed design, size S x D
        :param npStar: size of the LHD to be generated. N* >= N
        :param ndStar: number of translation steps for the seed in each dimension
        :return: rescaled seeds, size S x D
        """
        ns, nv = seed.shape
        if ns == 1:
            seed = np.ones((1, nv))
            return seed
        uf = ns * np.ones(nv)
        ut = ((npStar / ndStar) - ndStar * (nv - 1) + 1) * np.ones(nv)
        a = (ut - 1) / (uf - 1)
        b = ut - a * uf

        return np.round(a * seed + b)

    @staticmethod
    def _translate_propagate(seed, npStar, ndStar):
        """
        Translates and propagates the seed design to a LHD of size npStar (which might exceed the requested size N)

        :param seed: seed design, size S x D
        :param npStar: size of the LHD to be generated (N*). 
        :param ndStar: number of translation steps for the seed in each dimension
        :return: LHD data matrix, size N* x D (still to be shrinked).
        """
        nv = seed.shape[1]
        X = seed

        for c1 in range(0, nv):
            # Propagation step
            seed = X
            # Define translation
            d = np.concatenate((np.power(ndStar, c1 - 1) * np.ones(np.max((c1, 0))),
                                [npStar / ndStar],
                                np.power(ndStar, c1) * np.ones(nv - np.max((c1, 0)) - 1)))
            for c2 in np.arange(1, ndStar):
                # Translation steps
                seed = seed + d
                X = np.vstack((X, seed))

        assert (X.shape == (npStar, nv))
        return X

    @staticmethod
    def _shrink(X, npoints):
        """
        When designs are generated that are larger than the requested number of points (N* > N), resize them.
        If the size was correct all along, the LHD is returned unchanged.

        :param X: Generated LHD, size N* x D, with N* >= N
        :param npoints: What size to resize to (N)
        :return: LHD data matrix, size N x D
        """
        npStar, nv = X.shape

        # Pick N samples nearest to centre of X
        centre = npStar * np.ones((1, nv)) / 2.
        distances = cdist(X, centre).ravel()
        idx = np.argsort(distances)
        X = X[idx[:npoints], :]

        # Translate to origin
        X -= np.min(X, axis=0) - 1

        # Collapse gaps in the design to assure all cell projections onto axes have 1 sample
        Xs = np.argsort(X, axis=0)
        X[Xs, np.arange(nv)] = np.tile(np.arange(1, npoints + 1), (nv, 1)).T
        assert (X.shape[0] == npoints)
        return X
