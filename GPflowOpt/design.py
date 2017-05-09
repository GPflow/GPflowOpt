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


class Design(object):
    """
    Space-filling designs generated within a domain.
    """

    def __init__(self, size, domain):
        super(Design, self).__init__()
        self.size = size
        self.domain = domain

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
