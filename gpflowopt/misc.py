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
import pandas as pd
import gpflowopt.acquisition


def randomize_model(model):
    assert isinstance(model, gpflow.models.Model)
    for p in model.parameters:
        if p.trainable:
            prior = p.prior or gpflow.priors.Gaussian(0., 1.)  # if undefined, use standard normal
            rvalue = np.array(prior.sample(p.shape or (1,))) # Return values of GPflow priors sample() aren't consistent
            if not p.prior:
                rvalue = p.transform.forward(rvalue)
            p.assign(rvalue if len(p.shape) > 0 else rvalue.squeeze())


def hmc_eval(acquisition, Xcand, num_samples=100, gradients=True, lmin=5, lmax=20, epsilon=0.01, logprobs=False):
    assert isinstance(acquisition, gpflowopt.acquisition.Acquisition)
    init_state = acquisition.read_trainables()
    hmc = gpflow.train.HMC()
    hmc_opts = dict(lmin=lmin, lmax=lmax, epsilon=epsilon, logprobs=logprobs, num_samples=num_samples)
    samples = pd.concat([hmc.sample(m, **hmc_opts) for m in acquisition.models], axis=1)
    method = acquisition.evaluate_with_gradients if gradients else lambda X: (acquisition.evaluate(X),)
    evaluations = []
    for i, s in samples.iterrows():
        acquisition.assign(s)
        evaluations.append(method(Xcand))
    acquisition.assign(init_state)
    hmc_result = tuple(np.mean(np.stack(arg), axis=0) for arg in zip(*evaluations))
    return hmc_result if len(hmc_result) > 1 else hmc_result[0]

