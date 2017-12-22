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

from gpflow import DataHolder, autoflow, settings, params_as_tensors
from gpflow.core import TensorConverter
import numpy as np
from .transforms import LinearTransform, DataTransform
from .domain import UnitCube
from .models import ModelWrapper


class DataScaler(ModelWrapper):
    """
    Model-wrapping class, primarily intended to assure the data in GPflow models is scaled.

    One DataScaler wraps one GPflow model, and can scale the input as well as the output data. By default,
    if any kind of object attribute is not found in the datascaler object, it is searched on the wrapped model.

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
        # model sanity checks, slightly stronger conditions than the wrapper

        # Initial configuration of the datascaler & scale
        n_inputs = model.X.shape[1]
        t1 = (domain or UnitCube(n_inputs)) >> UnitCube(n_inputs)
        t1.compile()
        print(model.X.read_value())
        print(t1.forward(model.X.read_value()))
        model.X.assign(t1.forward(model.X.read_value()))
        print(model.X.read_value())
        t2 = DataScaler._compute_output_transform(model.Y, normalize_Y)
        t2.compile()
        model.Y.assign(t2.forward(model.Y.read_value()))

        # Final setup
        super(DataScaler, self).__init__(model)
        print(model.X.read_value())
        self.itf = t1
        self.otf = t2
        self._normalize_Y = normalize_Y

    @staticmethod
    def _compute_output_transform(value, scale):
        if isinstance(value, DataHolder):
            value = value.read_value()
        outdim = value.shape[1]
        if not scale:
            # Output normalization turned off. Reset transform to identity
            return LinearTransform(np.ones(outdim), np.zeros(outdim))
        else:
            return ~LinearTransform(value.std(axis=0), value.mean(axis=0))

    def _update_output_transform(self, value):
        t = DataScaler._compute_output_transform(value, self._normalize_Y)
        self.otf.assign(t)

    def set_input_transform(self, t):
        """
        Configure a new input transform.

        Data in the wrapped model is automatically updated with the new transform.

        :param t: :class:`.DataTransform` object: the new input transform.
        """
        assert isinstance(t, DataTransform)
        X = self.X.read_value()  # unscales the data
        self.itf.assign(t)
        self.X = X

    def set_output_transform(self, t):
        """
        Configure a new output transform. Data in the model is automatically updated with the new transform.
        
        :param t: :class:`.DataTransform` object: the new output transform.
        """
        assert isinstance(t, DataTransform)
        Y = self.Y.read_value()
        self.otf.assign(t)
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
        Y = self.Y.read_value()
        self._normalize_Y = flag
        self._update_output_transform(Y)
        self.Y = Y

    # Methods overwriting methods of the wrapped model.
    @property
    def X(self):
        """
        Returns the input data of the model, unscaled.

        :return: :class:`.DataHolder`: unscaled input data
        """
        if TensorConverter.tensor_mode(self):
            return self.itf.build_backward(self.wrapped.X)
        else:
            return DataHolder(self.itf.backward(self.wrapped.X.read_value()))

    @property
    def Y(self):
        """
        Returns the output data of the wrapped model, unscaled.

        :return: :class:`.DataHolder`: unscaled output data
        """
        if TensorConverter.tensor_mode(self):
            return self.otf.build_backward(self.wrapped.Y)
        else:
            return DataHolder(self.otf.backward(self.wrapped.Y.read_value()))

    @X.setter
    def X(self, x):
        """
        Set the input data. Applies the input transform before setting the data of the wrapped model.
        """
        self.wrapped.X = self.itf.forward(x.read_value if isinstance(x, DataHolder) else x)

    @Y.setter
    def Y(self, y):
        """
        Set the output data. In case normalize_Y=True, the appropriate output transform is updated. It is then
        applied on the data before setting the data of the wrapped model.
        """
        value = y.read_value() if isinstance(y, DataHolder) else y
        if self._normalize_Y:
            self._update_output_transform(value)
        self.wrapped.Y = self.otf.forward(value)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        build_predict builds the TensorFlow graph for prediction. Similar to the method in the wrapped model, however
        the input points are transformed using the input transform. The returned mean and variance are transformed
        backward using the output transform.
        """
        f, var = self.wrapped._build_predict(self.itf.build_forward(Xnew), full_cov=full_cov)
        return self.otf.build_backward(f), self.otf.build_backward_variance(var)

    @autoflow((settings.tf_float, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        return self._build_predict(Xnew)

    @autoflow((settings.tf_float, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        return self._build_predict(Xnew, full_cov=True)

    @autoflow((settings.tf_float, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        f, var = self.wrapped._build_predict(self.itf.build_forward(Xnew))
        f, var = self.likelihood.predict_mean_and_var(f, var)
        return self.otf.build_backward(f), self.otf.build_backward_variance(var)

    @autoflow((settings.tf_float, [None, None]), (settings.tf_float, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        """
        mu, var = self.wrapped._build_predict(self.itf.build_forward(Xnew))
        Ys = self.otf.build_forward(Ynew)
        return self.likelihood.predict_density(mu, var, Ys)
