# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gin
from ..nodeclassification.generator_wrappers import SbmGeneratorWrapper


@gin.configurable
class SSLSbmGeneratorWrapper(SbmGeneratorWrapper):

  def __init__(self, param_sampler_specs, marginal=False,
               normalize_features=True, marginal_params=[]):
    super(SSLSbmGeneratorWrapper, self).__init__(param_sampler_specs, marginal, normalize_features)
    self._marginal_params = marginal_params


  def SampleConfig(self, marginal=False):
    config = {}
    marginal_params = self._marginal_params
    if marginal and len(marginal_params) == 0:
      marginal_params.append(self._ChooseMarginalParam())
    fixed_params = []
    for param_name, spec in self._param_sampler_specs.items():
      param_value = None
      if marginal and len(marginal_params) > 0:
        # If the param is not a marginal param, give it its default (if possible)
        if param_name not in marginal_params:
          if spec.default_val is not None:
            fixed_params.append(param_name)
            param_value = spec.default_val
      # If the param val is still None, give it a random value.
      if param_value is None:
        param_value = spec.sampler_fn(spec)
      config[param_name] = param_value
    return config, marginal_params, fixed_params