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

from GPflow.param import DataHolder, AutoFlow, Parameterized
from GPflow.model import Model
from GPflow import settings
import numpy as np
from .transforms import LinearTransform
from .domain import ContinuousParameter

float_type = settings.dtypes.float_type


class Normalizer(Parameterized):
    def __init__(self, model, domain=None, normalize_output=True):
        assert (model is not None)
        assert (hasattr(model, 'X'))
        assert (hasattr(model, 'Y'))
        assert (hasattr(model, 'build_predict'))
        assert (isinstance(model, Model))
        self.wrapped = model
        super(Normalizer, self).__init__()
        n_inputs = model.X.shape[1]
        n_outputs = model.Y.shape[1]

        unitcube = np.sum([ContinuousParameter('x{0}'.format(i), 0, 1) for i in np.arange(n_inputs)])
        self._input_transform = domain >> unitcube if domain else LinearTransform(np.ones(n_inputs),
                                                                                  np.zeros(n_inputs))

        self._output = normalize_output
        self.output_transform = LinearTransform(np.ones(n_outputs), np.zeros(n_outputs))
        self.X = model.X.value
        self.Y = model.Y.value

    def __getattr__(self, item):
        return self.__getattribute__('wrapped').__getattribute__(item)

    def __setattr__(self, key, value):
        if key is 'wrapped':
            object.__setattr__(self, key, value)
            value.__setattr__('_parent', self)
            return

        super(Normalizer, self).__setattr__(key, value)

    def __eq__(self, other):
        return self.wrapped.__eq__(other)

    # Overwrites
    @property
    def X(self):
        return DataHolder(self.input_transform.backward(self.wrapped.X.value))

    @property
    def Y(self):
        return DataHolder(self.output_transform.backward(self.wrapped.Y.value))

    @X.setter
    def X(self, value):
        self.wrapped.X = self.input_transform.forward(value)

    @Y.setter
    def Y(self, value):
        if self._output:
            self.output_transform.assign(~LinearTransform(value.std(axis=0), value.mean(axis=0)))
        # self.highest_parent._kill_autoflow()
        self.wrapped.Y = self.output_transform.forward(value)

    @property
    def input_transform(self):
        return self._input_transform

    # @input_transform.setter
    # def input_transform(self, t):
    #    X = self.X.value
    #    self._input_transform.assign(t)
    #    self.X = X

    def build_predict(self, Xcand, full_cov=False):
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(Xcand), full_cov=full_cov)
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, X):
        return self.build_predict(X)

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, X):
        return self.build_predict(X, full_cov=True)

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, X):
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(X))
        f, var = self.wrapped.likelihood.predict_mean_and_var(f, var)
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)
