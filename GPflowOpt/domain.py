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
from itertools import chain

from GPflow.param import Parentable


class Domain(Parentable):
    """
    Basic class, representing an optimization domain by aggregating several parameters.
    """

    def __init__(self, parameters):
        super(Domain, self).__init__()
        self._parameters = parameters

    @property
    def lower(self):
        return np.array(list(map(lambda param: param.lower, self._parameters))).flatten()

    @property
    def upper(self):
        return np.array(list(map(lambda param: param.upper, self._parameters))).flatten()

    # def optimize(self, optimizer, objectivefx):
    #    optimizer.domain = self
    #    result = optimizer.optimize(objectivefx)

    def __add__(self, other):
        assert isinstance(other, Domain)
        return Domain(self._parameters + other._parameters)

    @property
    def size(self):
        return sum(map(lambda param: param.size, self._parameters))

    def __setattr__(self, key, value):
        super(Domain, self).__setattr__(key, value)
        if key is not '_parent':
            if isinstance(value, Parentable):
                value._parent = self
            if isinstance(value, list):
                for val in (x for x in value if isinstance(x, Parentable)):
                    val._parent = self

    def __eq__(self, other):
        return self._parameters == other._parameters

    def __contains__(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] is not self.size:
            return False
        return np.all(np.logical_and((self.lower <= X), (X <= self.upper)))

    def __iter__(self):
        for v in chain(*map(iter, self._parameters)):
            yield v

    def __getitem__(self, item):
        return self._parameters[item]

    @property
    def value(self):
        return np.vstack(map(lambda p: p.value, self._parameters)).T

    @value.setter
    def value(self, x):
        x = np.atleast_2d(x)
        assert (len(x.shape) == 2)
        assert (x.shape[1] == self.size)
        offset = 0
        for p in self._parameters:
            p.value = x[:, offset:offset + p.size]
            offset += p.size


class Parameter(Domain):
    """
    Abstract class representing a parameter (which corresponds to a one-dimensional domain)
    This class can be derived for continuous, discrete and categorical parameters
    """

    def __init__(self, label, xinit):
        super(Parameter, self).__init__([self])
        self.label = label
        self._x = np.atleast_1d(xinit)

    @Domain.size.getter
    def size(self):
        return 1

    def __iter__(self):
        yield self

    @Domain.value.getter
    def value(self):
        return self._x

    @value.setter
    def value(self, x):
        x = np.atleast_1d(x)
        self._x = x.ravel()


class ContinuousParameter(Parameter):
    def __init__(self, label, lb, ub, xinit=None):
        self._range = [lb, ub]
        super(ContinuousParameter, self).__init__(label, xinit or ((ub + lb) / 2.0))

    @Parameter.lower.getter
    def lower(self):
        return self._range[0]

    @Parameter.upper.getter
    def upper(self):
        return self._range[1]

    @lower.setter
    def lower(self, value):
        self._range[0] = value

    @upper.setter
    def upper(self, value):
        self._range[1] = value

    def __eq__(self, other):
        return isinstance(other, ContinuousParameter) and self.lower == other.lower and self.upper == other.upper
