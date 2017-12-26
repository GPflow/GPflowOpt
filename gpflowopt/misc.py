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

import gpflow
import numpy as np


def randomize_model(model):
    assert isinstance(model, gpflow.models.Model)
    for p in model.parameters:
        if p.trainable:
            prior = p.prior or gpflow.priors.Gaussian(0., 1.)  # if undefined, use standard normal
            rvalue = np.array(prior.sample(p.shape or (1,))) # Return values of GPflow priors sample() aren't consistent
            p.assign(rvalue if len(p.shape) > 0 else rvalue.squeeze())
