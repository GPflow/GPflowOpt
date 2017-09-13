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
from gpflow.param import Parentable

from .transforms import LinearTransform


class Domain(Parentable):
    """
    A domain representing the mathematical space over which is optimized.
    """

    def __init__(self, parameters):
        super(Domain, self).__init__()
        self._parameters = parameters

    @property
    def lower(self):
        """
        Lower bound of the domain, corresponding to a numpy array with the lower value of each parameter
        """
        return np.array(list(map(lambda param: param.lower, self._parameters))).flatten()

    @property
    def upper(self):
        """
        Upper bound of the domain, corresponding to a numpy array with the upper value of each parameter
        """
        return np.array(list(map(lambda param: param.upper, self._parameters))).flatten()

    def __add__(self, other):
        assert isinstance(other, Domain)
        return Domain(self._parameters + other._parameters)

    @property
    def size(self):
        """
        Returns the dimensionality of the domain
        """
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
        return np.all(np.logical_and(np.logical_or(self.lower < X, np.isclose(self.lower, X)),
                                     np.logical_or(X < self.upper, np.isclose(self.upper, X))))

    def __iter__(self):
        for v in chain(*map(iter, self._parameters)):
            yield v

    def __getitem__(self, items):
        if isinstance(items, list):
            return np.sum([self[item] for item in items])

        if isinstance(items, str):
            labels = [param.label for param in self._parameters]
            items = labels.index(items)

        return self._parameters[items]

    def __rshift__(self, other):
        assert(self.size == other.size)
        A = (other.upper - other.lower) / (self.upper - self.lower)
        b = -self.upper * A + other.upper
        return LinearTransform(A, b)

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

    def _repr_html_(self):
        """
        Build html string for table display in jupyter notebooks.
        """
        html = ["<table id='domain' width=100%>"]

        # Table header
        columns = ['Name', 'Type', 'Values']
        header = "<tr>"
        header += ''.join(map(lambda l: "<td>{0}</td>".format(l), columns))
        header += "</tr>"
        html.append(header)

        # Add parameters
        html.append(self._html_table_rows())
        html.append("</table>")

        return ''.join(html)

    def _html_table_rows(self):
        return ''.join(map(lambda l: l._html_table_rows(), self._parameters))


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
        """
        One parameter has a dimensionality of 1
        :return: 1
        """
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

    def _html_table_rows(self):
        """
        Html row representation of a Parameter. Should be overwritten in subclasses objects.
        """
        return "<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>".format(self.label, 'N/A', 'N/A')


class ContinuousParameter(Parameter):
    def __init__(self, label, lb, ub, xinit=None):
        self._range = np.array([lb, ub], dtype=float)
        super(ContinuousParameter, self).__init__(label, xinit or ((ub + lb) / 2.0))

    @Parameter.lower.getter
    def lower(self):
        return np.array([self._range[0]])

    @Parameter.upper.getter
    def upper(self):
        return np.array([self._range[1]])

    @lower.setter
    def lower(self, value):
        self._range[0] = value

    @upper.setter
    def upper(self, value):
        self._range[1] = value

    def __eq__(self, other):
        return isinstance(other, ContinuousParameter) and self.lower == other.lower and self.upper == other.upper

    def _html_table_rows(self):
        """
        Html row representation of a ContinuousParameter.
        """
        return "<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>".format(self.label, 'Continuous', str(self._range))


class UnitCube(Domain):
    """
    The unit domain [0, 1]^d
    """
    def __init__(self, n_inputs):
        params = [ContinuousParameter('u{0}'.format(i), 0, 1) for i in np.arange(n_inputs)]
        super(UnitCube, self).__init__(params)
