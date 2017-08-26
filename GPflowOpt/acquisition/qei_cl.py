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


from .ei import ExpectedImprovement
import numpy as np


class QEI_CL(ExpectedImprovement):
    """
    This class is an implementation of the constant liar heuristic (min case)
    for using Expected Improvement in the batch case.

    See:
    Ginsbourger D., Le Riche R., Carraro L. (2010)
    Kriging Is Well-Suited to Parallelize Optimization.
    """

    def __init__(self, model, batch_size):
        super(QEI_CL, self).__init__(model, batch_size=batch_size)
        self.in_batch = False

    def set_batch(self, *args):
        assert self.in_batch, 'Set batch must be called within a context'

        X = np.vstack((self.X_,) + args)
        Y = np.vstack((self.Y_,) + (self.fmin.value,)*len(args))
        self.set_data(X, Y)

    def __enter__(self):
        self.in_batch = True

        # Save original dataset of the model
        self.X_, self.Y_ = np.copy(self.data[0]), np.copy(self.data[1])

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original dataset of the model
        self.set_data(self.X_, self.Y_)

        self.in_batch = False
