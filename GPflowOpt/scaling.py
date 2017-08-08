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
from GPflow.model import Model, GPModel
from GPflow import settings
import numpy as np
from .transforms import LinearTransform, DataTransform
from .domain import UnitCube

float_type = settings.dtypes.float_type


class DataScaler(GPModel):
    """
    Model-wrapping class, primarily intended to assure the data in GPflow models is scaled. One DataScaler wraps one
    GPflow model, and can scale the input as well as the output data. By default, if any kind of object attribute
    is not found in the datascaler object, it is searched on the wrapped model.

    The datascaler supports both input as well as output scaling, although both scalings are set up differently:

    - For input, the transform is not automatically generated. By default, the input transform is the identity
      transform. The input transform can be set through the setter property, or by specifying a domain in the
      constructor. For the latter, the input transform will be initialized as the transform from the specified domain to
      a unit cube. When X is updated, the transform does not change.

    - If enabled: for output the data is always scaled to zero mean and unit variance. This means that if the Y property
      is set, the output transform is first calculated, then the data is scaled.


    By default, :class:`~.acquisition.Acquisition` objects will always wrap each model received. However, the input and output transforms
    will be the identity transforms, and output normalization is switched off. It is up to the user (or
    specialized classes such as the BayesianOptimizer) to correctly configure the datascalers involved.

    By carrying out the scaling at such a deep level in the framework, it is possible to keep the scaling
    hidden throughout the rest of GPflowOpt. This means that, during implementation of acquisition functions it is safe
    to assume the data is not scaled, and is within the configured optimization domain. There is only one exception:
    the hyperparameters are determined on the scaled data, and are NOT automatically unscaled by this class because the
    datascaler does not know what model is wrapped and what kernels are used. Should hyperparameters of the model be
    required, it is the responsibility of the implementation to rescale the hyperparameters. Additionally, applying
    hyperpriors should anticipate for the scaled data.
    """
    def __init__(self, model, domain=None, normalize_Y=False):
        """
        :param model: model to be wrapped
        :param domain: (default: None) if supplied, the input transform is configured from the supplied domain to
            :class:`.UnitCube`. If None, the input transform defaults to the identity transform.
        :param normalize_Y: (default: False) enable automatic scaling of output values to zero mean and unit
         variance.
        """
        # model sanity checks
        assert (model is not None)
        assert (isinstance(model, GPModel))
        self._parent = None

        # Wrap model
        self.wrapped = model

        # Initial configuration of the datascaler
        n_inputs = model.X.shape[1]
        n_outputs = model.Y.shape[1]
        self._input_transform = (domain or UnitCube(n_inputs)) >> UnitCube(n_inputs)
        self._normalize_Y = normalize_Y
        self._output_transform = LinearTransform(np.ones(n_outputs), np.zeros(n_outputs))

        # The assignments in the constructor of GPModel take care of initial re-scaling of model data.
        super(DataScaler, self).__init__(model.X.value, model.Y.value, None, None, 1, name=model.name+"_datascaler")
        del self.kern
        del self.mean_function
        del self.likelihood

    def __getattr__(self, item):
        """
        If an attribute is not found in this class, it is searched in the wrapped model
        """
        return self.wrapped.__getattribute__(item)

    def __setattr__(self, key, value):
        """
        If setting :attr:`wrapped` attribute, point parent to this object (the datascaler)
        """
        if key is 'wrapped':
            object.__setattr__(self, key, value)
            value.__setattr__('_parent', self)
            return

        super(DataScaler, self).__setattr__(key, value)

    def __eq__(self, other):
        return self.wrapped == other

    def __str__(self, prepend=''):
        return self.wrapped.__str__(prepend)

    @property
    def input_transform(self):
        """
        Get the current input transform
        
        :return: :class:`.DataTransform` input transform object
        """
        return self._input_transform

    @input_transform.setter
    def input_transform(self, t):
        """
        Configure a new input transform. Data in the model is automatically updated with the new transform.
        
        :param t: :class:`.DataTransform` object: the new input transform.
        """
        assert(isinstance(t, DataTransform))
        X = self.X.value
        self._input_transform.assign(t)
        self.X = X

    @property
    def output_transform(self):
        """
        Get the current output transform
        
        :return: :class:`.DataTransform` output transform object
        """
        return self._output_transform

    @output_transform.setter
    def output_transform(self, t):
        """
        Configure a new output transform. Data in the model is automatically updated with the new transform.
        
        :param t: :class:`.DataTransform` object: the new output transform.
        """
        assert (isinstance(t, DataTransform))
        Y = self.Y.value
        self._output_transform.assign(t)
        self.Y = Y

    @property
    def normalize_output(self):
        """
        :return: boolean, indicating if output is automatically scaled to zero mean and unit variance.
        """
        return self._normalize_Y

    @normalize_output.setter
    def normalize_output(self, flag):
        """
        Enable/disable automated output scaling. If switched off, the output transform becomes the identity transform.
        If enabled, data will be automatically scaled to zero mean and unit variance. When the output normalization is
        switched on or off, the data in the model is automatically adapted.
        
        :param flag: boolean, turn output scaling on or off
        """

        self._normalize_Y = flag
        if not flag:
            # Output normalization turned off. Reset transform to identity
            self.output_transform = LinearTransform(np.ones(self.Y.value.shape[1]), np.zeros(self.Y.value.shape[1]))
        else:
            # Output normalization enabled. Trigger scaling.
            self.Y = self.Y.value

    # Methods overwriting methods of the wrapped model.
    @property
    def X(self):
        """
        Returns the input data of the model, unscaled.

        :return: :class:`.DataHolder`: unscaled input data
        """
        return DataHolder(self.input_transform.backward(self.wrapped.X.value))

    @property
    def Y(self):
        """
        Returns the output data of the wrapped model, unscaled.

        :return: :class:`.DataHolder`: unscaled output data
        """
        return DataHolder(self.output_transform.backward(self.wrapped.Y.value))

    @X.setter
    def X(self, x):
        """
        Set the input data. Applies the input transform before setting the data of the wrapped model.
        """
        self.wrapped.X = self.input_transform.forward(x.value if isinstance(x, DataHolder) else x)

    @Y.setter
    def Y(self, y):
        """
        Set the output data. In case normalize_Y=True, the appropriate output transform is updated. It is then
        applied on the data before setting the data of the wrapped model.
        """
        value = y.value if isinstance(y, DataHolder) else y
        if self.normalize_output:
            self.output_transform.assign(~LinearTransform(value.std(axis=0), value.mean(axis=0)))
        self.wrapped.Y = self.output_transform.forward(value)

    def build_predict(self, Xnew, full_cov=False):
        """
        build_predict builds the TensorFlow graph for prediction. Similar to the method in the wrapped model, however
        the input points are transformed using the input transform. The returned mean and variance are transformed
        backward using the output transform.
        """
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(Xnew), full_cov=full_cov)
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        f, var = self.wrapped.build_predict(self.input_transform.build_forward(Xnew))
        f, var = self.likelihood.predict_mean_and_var(f, var)
        return self.output_transform.build_backward(f), self.output_transform.build_backward_variance(var)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        """
        mu, var = self.build_predict(Xnew)
        Ys = self.output_transform.build_forward(Ynew)
        return self.likelihood.predict_density(mu, var, Ys)
