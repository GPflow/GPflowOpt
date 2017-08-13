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

import tensorflow as tf
from GPflow import settings

float_type = settings.dtypes.float_type


def rowwise_gradients(Y, X):
    """
    For a 2D Tensor Y, compute the derivative of each columns w.r.t  a 2D tensor X.

    This is done with while_loop, because of a known incompatibility between map_fn and gradients.
    """
    num_rows = tf.shape(Y)[0]
    num_feat = tf.shape(X)[0]

    def body(old_grads, row):
        g = tf.expand_dims(tf.gradients(Y[row], X)[0], axis=0)
        new_grads = tf.concat([old_grads, g], axis=0)
        return new_grads, row + 1

    def cond(_, row):
        return tf.less(row, num_rows)

    shape_invariants = [tf.TensorShape([None, None]), tf.TensorShape([])]
    grads, _ = tf.while_loop(cond, body, [tf.zeros([0, num_feat], float_type), tf.constant(0)],
                             shape_invariants=shape_invariants)

    return grads