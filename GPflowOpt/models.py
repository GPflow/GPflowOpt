# Copyright 2017 Joachim van der Herten, Nicolas Knudde
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
from .tf_wraps import rowwise_gradients

from GPflow.param import Parameterized, AutoFlow
from GPflow.model import Model, GPModel
from GPflow.likelihoods import Gaussian
from GPflow import settings

import tensorflow as tf

float_type = settings.dtypes.float_type


class ModelWrapper(Parameterized):
    """
    Modelwrapper class
    """
    def __init__(self, model):
        """
        :param model: model to be wrapped
        """
        super(ModelWrapper, self).__init__()

        # Wrap model
        assert isinstance(model, (Model, ModelWrapper))
        self.wrapped = model

    def __getattr__(self, item):
        """
        If an attribute is not found in this class, it is searched in the wrapped model
        """

        # Exception for AF storages, if a method with the same name exists in this class, do not find the cache
        # in the wrapped model.
        if item.endswith('_AF_storage'):
            method = item[1:].rstrip('_AF_storage')
            if method in dir(self):
                raise AttributeError("{0} has no attribute {1}".format(self.__class__.__name__, item))

        return getattr(self.wrapped, item)

    def __setattr__(self, key, value):
        """
        If setting :attr:`wrapped` attribute, point parent to this object (the datascaler)
        """
        if key is 'wrapped':
            object.__setattr__(self, key, value)
            value.__setattr__('_parent', self)
            return

        if key is '_needs_recompile':
            setattr(self.wrapped, key, value)
            return

        super(ModelWrapper, self).__setattr__(key, value)

    def __eq__(self, other):
        return self.wrapped == other

    def __str__(self, prepend=''):
        return self.wrapped.__str__(prepend)


class MGP(ModelWrapper):
    """
    Marginalisation of the hyperparameters during evaluation time using a Laplace Approximation
    Key reference:

    ::

       @article{Garnett:2013,
          title={Active learning of linear embeddings for Gaussian processes},
          author={Garnett, Roman and Osborne, Michael A and Hennig, Philipp},
          journal={arXiv preprint arXiv:1310.6740},
          year={2013}
        }
    """

    def __init__(self, model):
        assert isinstance(model, GPModel), "Object has to be a GP model"
        assert isinstance(model.likelihood, Gaussian), "Likelihood has to be Gaussian"
        super(MGP, self).__init__(model)

    def build_predict(self, fmean, fvar, theta):
        h = tf.hessians(self.build_likelihood() + self.build_prior(), theta)[0]
        L = tf.cholesky(-h)

        N = tf.shape(fmean)[0]
        D = tf.shape(fmean)[1]

        fmeanf = tf.reshape(fmean, [N * D, 1])      # N*D x 1
        fvarf = tf.reshape(fvar, [N * D, 1])        # N*D x 1

        Dfmean = rowwise_gradients(fmeanf, theta)   # N*D x k
        Dfvar = rowwise_gradients(fvarf, theta)     # N*D x k

        tmp1 = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(Dfmean)))    # N*D x k
        tmp2 = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(Dfvar)))     # N*D x k
        return fmean, 4 / 3 * fvar + tf.reshape(tf.reduce_sum(tf.square(tmp1), axis=1), [N, D]) \
               + 1 / 3 / (fvar + 1E-3) * tf.reshape(tf.reduce_sum(tf.square(tmp2), axis=1), [N, D])

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        theta = self._predict_f_AF_storage['free_vars']
        fmean, fvar = self.wrapped.build_predict(Xnew)
        return self.build_predict(fmean, fvar, theta)

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        theta = self._predict_y_AF_storage['free_vars']
        pred_f_mean, pred_f_var = self.wrapped.build_predict(Xnew)
        fmean, fvar = self.wrapped.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
        return self.build_predict(fmean, fvar, theta)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        theta = self._predict_density_AF_storage['free_vars']
        pred_f_mean, pred_f_var = self.wrapped.build_predict(Xnew)
        pred_f_mean, pred_f_var = self.build_predict(pred_f_mean, pred_f_var, theta)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
