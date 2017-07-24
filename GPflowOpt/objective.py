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
from functools import wraps


def batch_apply(fun):
    @wraps(fun)
    def batch_wrapper(X):
        responses = (fun(x) for x in np.atleast_2d(X))
        sep = tuple(zip(*(r if isinstance(r, tuple) else (r,) for r in responses)))
        f = np.vstack(sep[0])
        if len(sep) == 1:
            return f

        g_stacked = np.stack((np.atleast_2d(r) for r in sep[1]), axis=0)  # n x p x d
        print('results: ')
        print(sep[1])
        g = np.squeeze(g_stacked, axis=1)  # in case p = 1
        return f, g

    return batch_wrapper


def to_args(fun):
    @wraps(fun)
    def args_wrapper(X):
        X = np.atleast_2d(X)
        return fun(*X.T)

    return args_wrapper
