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

from collections import namedtuple

import numpy as np
import tensorflow as tf


class Data(namedtuple('Data', ['X', 'Y'])):
    """
    Corresponds to the input data X which is the same for every model,
    and column-wise concatenation of the Y data over all models
    """

    @classmethod
    def create(cls, models, tf_mode=False):
        """
        Given an iterable container of models, condense their contained training data into a Data object.
        :param models: iterable of models
        :param tf_mode: (default: False) is the object requested in tf_mode or not?
        :return: Data object
        """
        if tf_mode:
            X, Y = models[0].X, tf.concat(list(map(lambda model: model.Y, models)), 1)
        else:
            X, Y = models[0].X.value, np.hstack(map(lambda model: model.Y.value, models))
        return cls(X=X, Y=Y)

    def save(self, fname):
        """
        Dump the data to a file (used for the failsafe context of BayesianOptimizer)
        :param fname: path to file to save data to.
        """
        np.savez(fname, **self._asdict())

    def update_model(self, model, Ycols=None):
        """
        Update the data of a given model.
        :param model: model to be updated.
        :param Ycols: (optional) columns of output data (Y) to be passed into the model. If unspecified, the first P
         columns are used, with P equal to the current number of outputs of the model.
        """
        model.X = self.X
        Ycols = np.arange(self.Y.shape[1]) if Ycols is None else Ycols
        model.Y = self.Y[:, np.atleast_1d(Ycols)]

    def add(self, Xn, Yn):
        """
        Given new X/Y, add them to the data object. The second dimension of Xn and Yn must match those of X/Y in Data
        :param Xn: New input samples. 2D ndarray n x d
        :param Yn: New output samples. 2D ndarray n x p
        :return: new Data object, containing the old data augmented with the new data.
        """
        assert self.X.shape[1] == Xn.shape[-1]
        assert self.Y.shape[1] == Yn.shape[-1]
        X = np.vstack((self.X, Xn))
        Y = np.vstack((self.Y, Yn))
        return self.__class__(X=X, Y=Y)