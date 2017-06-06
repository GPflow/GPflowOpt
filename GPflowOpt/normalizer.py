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

from GPflow.param import DataHolder, AutoFlow
from GPflow.model import Model
from GPflow import settings
import numpy as np
from .transforms import LinearTransform
from .domain import ContinuousParameter

float_type = settings.dtypes.float_type


class Normalizer(Model):
    def __init__(self, model, domain, scale_input=True, normalize_output=True):
        assert (model is not None)
        assert (hasattr(model, 'X'))
        assert (hasattr(model, 'Y'))
        assert (hasattr(model, 'build_predict'))
        assert (isinstance(model, Model))
        self.wrapped = model
        super(Normalizer, self).__init__()

        self._input = scale_input
        unitcube = np.sum([ContinuousParameter('x{0}'.format(i), 0, 1) for i in np.arange(domain.size)])
        self.input_transform = domain >> (unitcube if self._input else domain)

        self._output = normalize_output
        self.output_transform = LinearTransform(np.ones(model.Y.shape[1]), np.zeros(model.Y.shape[1]))

        # Initial normalization (if enabled) happens here (these lines invoke the setter properties)
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
            self.output_transform = ~LinearTransform(value.std(axis=0), value.mean(axis=0))
        self.wrapped.Y = self.output_transform.forward(value)

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, X):
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(X))
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, X):
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(X), full_cov=True)
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, X):
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(X))
        f, var = self.wrapped.likelihood.predict_mean_and_var(f, var)
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)
