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
from GPflowOpt.domain import ContinuousParameter


class Design(object):
    """
    Space-filling designs generated within a domain.
    """

    def __init__(self, size, domain):
        super(Design, self).__init__()
        self.size = size
        self.domain = domain

    def generate(self):
        """
        Returns a design of the requested size (N) and domain dimensionality (D). All data points are contained
        by the design.
        :return: 2D ndarray, N x D
        """
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


class LatinHyperCube(Design):
    """
    Latin hypercube with optimized maximin distance. Created with the Translational Propagation algorithm to obtain 
    some speed and avoid lengthy generation procedures. For dimensions smaller or equal to 6, this algorithm finds 
    the optimal LHD (or gets very close) with overwhelming probability. 
    Beyond 6D, this property fades, although the resulting designs are still acceptable. Somewhere beyond 15D this 
    algorithm tends to slow down a lot. Key reference is
    
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

    """

    def __init__(self, size, domain, max_seed_size=None):
        super(LatinHyperCube, self).__init__(size, domain)
        self._max_seed_size = max_seed_size or domain.size

    def generate(self):
        # Generate several TPLHDs with growing seed, select the one with the best intersite distance
        candidates = []
        scores = []

        for i in np.arange(1, min(self.size, self._max_seed_size) + 1):
            if i < 3:
                # Hardcoded seeds for 1 or two points.
                seed = np.arange(1, i + 1)[:, None] * np.ones((1, self.domain.size))
            else:
                # Generate larger seeds recursively by creating small LHD's
                seed = self._tplhs_design(i, np.ones((1, self.domain.size)))

            # Create all designs and compute score
            X = self._tplhs_design(self.size, seed)
            candidates.append(X)
            scores.append(np.min(pdist(candidates[-1])))

        # Transform best design (highest score) to specified domain
        cube = np.sum([ContinuousParameter('x{0}'.format(i), 1, self.size) for i in np.arange(self.domain.size)])
        transform = cube >> self.domain
        return np.clip(transform.forward(candidates[np.argmax(scores)]), self.domain.lower, self.domain.upper)

    def _tplhs_design(self, npoints, seed):
        """
        Creates a LHD with the Translational propagation algorithm with specified seed and design size
        :param npoints: size of design to generate (N), may differ from self.size.
        :param seed: 2D ndarray, the seed to use. S x D
        :return: LHD, 2D ndarray. N x D
        """
        ns, nv = seed.shape
        nd = np.power(npoints / float(ns), 1 / float(nv))
        ndStar = np.ceil(nd)

        # Determine npStar, the amount of points we'll be generating
        npStar = np.power(ndStar, nv) * ns if ndStar > nd else npoints

        # First assert scale of seed, then generate
        seed = self._rescale_seed(seed, npStar, ndStar)
        X = self._create(seed, npStar, ndStar)

        # In case the generated design is too big (as specified by npStar), get rid of some points
        return self._resize(X, npoints)

    @staticmethod
    def _rescale_seed(seed, npStar, ndStar):
        """
        Rescales the seeding pattern
        :param seed: 2D ndarray, S x D
        :param npStar: size of the LHD to be generated. N* >= N
        :param ndStar: number of translation steps for the seed in each dimension
        :return: rescaled seed, 2D ndarray, S x D
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
    def _create(seed, npStar, ndStar):
        """
        Creates an LHD given the rescaled seed
        :param seed: seed pattern, 2D ndarray S x D
        :param npStar: size of the LHD to be generated (N*). 
        :param ndStar: number of translation steps for the seed in each dimension
        :return: LHD, 2D ndarray N* x D which may be in need of resizing.
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
    def _resize(X, npoints):
        """
        When designs are generated that are larger than the requested number of points (N* > N), resize. 
        If the size was correct all along, the LHD is returned unchanged.
        :param X: Generated LHD, N* x D, with N* >= N
        :param npoints: What size to resize to (N)
        :return: LHD, 2D ndarray N x D
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
