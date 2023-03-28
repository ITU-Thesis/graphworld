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