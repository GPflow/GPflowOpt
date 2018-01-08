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


class ObjectiveWrapper(object):
    """
    A wrapper for objective functions.
    
    Filters out gradient information if necessary and keeps a count of the number of function evaluations.
    """
    def __init__(self, objective, exclude_gradient):
        super(ObjectiveWrapper, self).__init__()
        self._no_gradient = exclude_gradient
        self.counter = 0
        self._objective = objective
        self._previous_x = None

    def __call__(self, x):
        x = np.atleast_2d(x)
        f, g = self._objective(x)
        g_is_fin = np.isfinite(g)

        if np.all(g_is_fin):
            self._previous_x = x  # store the last known good value
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")

        self.counter += x.shape[0]
        if self._no_gradient:
            return f
        return f, g

