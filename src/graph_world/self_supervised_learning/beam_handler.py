import json
import math

from abc import ABC, abstractmethod
import apache_beam as beam
import gin
import numpy as np

from ..models.utils import ComputeNumPossibleConfigs, SampleModelConfig, GetCartesianProduct
from ..nodeclassification.beam_handler import NodeClassificationBeamHandler
from ..beam.benchmarker import BenchmarkGNNParDo


class BenchmarkGNNParDoSSL(BenchmarkGNNParDo):
  def __init__(self, benchmarker_wrappers, num_tuning_rounds, tuning_metric,
               tuning_metric_is_loss=False, save_tuning_results=False):
    super().__init__(benchmarker_wrappers, num_tuning_rounds, tuning_metric,
               tuning_metric_is_loss, save_tuning_results)
    self._pretext_tasks = [benchmarker_wrapper().GetPretextTasks() for
                           benchmarker_wrapper in benchmarker_wrappers]

  def process(self, element):
    output_data = {}
    output_data.update(element['generator_config'])
    output_data['marginal_param'] = element['marginal_param']
    output_data['fixed_params'] = element['fixed_params']
    output_data.update(element['metrics'])
    output_data['skipped'] = element['skipped']
    if 'target' in element:
      output_data['target'] = element['target']
    output_data['sample_id'] = element['sample_id']

    if element['skipped']:
      yield json.dumps(output_data)

    # for benchmarker in self._benchmarkers:
    for (benchmarker_class,
         benchmark_params,
         model_class,
         h_params,
         pretext_tasks) in zip(self._benchmarker_classes,
                          self._benchmark_params,
                          self._model_classes,
                          self._h_params,
                          self._pretext_tasks):
      if pretext_tasks is None:
        print(f'Running {benchmarker_class} and model f{model_class}')
      else:
        print(f'Running {benchmarker_class} and model f{model_class} with pretext tasks: {pretext_tasks}')

      num_possible_configs = ComputeNumPossibleConfigs(benchmark_params, h_params)
      num_tuning_rounds = min(num_possible_configs, self._num_tuning_rounds)

      if num_tuning_rounds == 1 or self._tuning_metric == '':
        benchmark_params_sample, h_params_sample = SampleModelConfig(benchmark_params,
                                                                     h_params)
        benchmarker = benchmarker_class(element['generator_config'],
                                        model_class,
                                        benchmark_params_sample,
                                        h_params_sample,
                                        pretext_tasks)
        benchmarker_out = benchmarker.Benchmark(element,
                                                tuning_metric=self._tuning_metric,
                                                tuning_metric_is_loss=self._tuning_metric_is_loss)
        val_metrics = benchmarker_out['val_metrics']
        test_metrics = benchmarker_out['test_metrics']

      else:
        configs = []
        val_metrics_list = []
        test_metrics_list = []
        full_product = False
        if num_tuning_rounds == 0:
          num_tuning_rounds = 1
          if benchmark_params is None:
            benchmark_params_product = []
          else:
            benchmark_params_product = list(GetCartesianProduct(benchmark_params))
          num_benchmark_configs = len(benchmark_params_product)
          if num_benchmark_configs > 0:
            num_tuning_rounds *= num_benchmark_configs
          if h_params is None:
            h_params_product = []
          else:
            h_params_product = list(GetCartesianProduct(h_params))
          num_h_configs = len(h_params_product)
          if num_h_configs > 0:
            num_tuning_rounds *= num_h_configs
          full_product = True
        for i in range(num_tuning_rounds):
          if full_product:
            if num_benchmark_configs > 0:
              benchmark_index = math.floor(i / num_h_configs)
              benchmark_params_sample = benchmark_params_product[benchmark_index]
            else:
              benchmark_params_sample = None
            if num_h_configs > 0:
              h_index = i % num_h_configs
              h_params_sample = h_params_product[h_index]
            else:
              h_params_sample = None
          else:
            benchmark_params_sample, h_params_sample = SampleModelConfig(benchmark_params,
                                                                         h_params)
          benchmarker = benchmarker_class(element['generator_config'],
                                          model_class,
                                          benchmark_params_sample,
                                          h_params_sample,
                                          pretext_tasks)
          benchmarker_out = benchmarker.Benchmark(element,
                                                  tuning_metric=self._tuning_metric,
                                                  tuning_metric_is_loss=self._tuning_metric_is_loss)
          configs.append((benchmark_params_sample, h_params_sample, pretext_tasks))
          val_metrics_list.append(benchmarker_out['val_metrics'])
          test_metrics_list.append(benchmarker_out['test_metrics'])

        val_scores = [metrics[self._tuning_metric] for metrics in val_metrics_list]
        test_scores = [metrics[self._tuning_metric] for metrics in test_metrics_list]
        if self._tuning_metric_is_loss:
          best_tuning_round = np.argmin(val_scores)
        else:
          best_tuning_round = np.argmax(val_scores)
        benchmark_params_sample, h_params_sample = configs[best_tuning_round]

        output_data['%s_%s_num_tuning_rounds' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskNames())] = num_tuning_rounds
        if self._save_tuning_results:
          output_data['%s_%s_configs' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskNames())] = configs
          output_data['%s_%s_val_scores' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskNames())] = val_scores
          output_data['%s_%s_test_scores' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskNames())] = test_scores

        val_metrics = val_metrics_list[best_tuning_round]
        test_metrics = test_metrics_list[best_tuning_round]

      # Return benchmark data for next beam stage.

      for key, value in val_metrics.items():
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskNames()}_val_{key}'] = value
      for key, value in test_metrics.items():
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskNames()}_test_{key}'] = value


      if benchmark_params_sample is not None:
        for key, value in benchmark_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskNames()}_train_{key}'] = value

      if h_params_sample is not None:
        for key, value in h_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskNames()}_model_{key}'] = value

    yield json.dumps(output_data)





@gin.configurable
class NodeClassificationBeamHandlerSSL(NodeClassificationBeamHandler):

  @gin.configurable
  def __init__(self, benchmarker_wrappers, generator_wrapper,
               num_tuning_rounds=1, tuning_metric='',
               tuning_metric_is_loss=False, ktrain=5, ktuning=5,
               save_tuning_results=False):
    super().__init__(benchmarker_wrappers, generator_wrapper,
               num_tuning_rounds=1, tuning_metric='',
               tuning_metric_is_loss=False, ktrain=5, ktuning=5,
               save_tuning_results=False)

    self._benchmark_par_do = BenchmarkGNNParDoSSL(
        benchmarker_wrappers, num_tuning_rounds, tuning_metric,
        tuning_metric_is_loss, save_tuning_results)