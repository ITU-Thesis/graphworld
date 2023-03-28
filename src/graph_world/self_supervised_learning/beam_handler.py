import json
import math

from abc import ABC, abstractmethod
import apache_beam as beam
import gin
import numpy as np

from .hparam_utils import ComputeNumPossibleConfigs, SampleModelConfig, GetCartesianProduct
from ..nodeclassification.beam_handler import NodeClassificationBeamHandler
from ..beam.benchmarker import BenchmarkGNNParDo
import random

class BenchmarkGNNParDoSSL(BenchmarkGNNParDo):
  def __init__(self, benchmarker_wrappers, num_tuning_rounds, tuning_metric,
               tuning_metric_is_loss=False, save_tuning_results=False, save_training_curves=False,
               sample_pretext_without_replacement=False):
    super().__init__(benchmarker_wrappers, num_tuning_rounds, tuning_metric,
               tuning_metric_is_loss, save_tuning_results)
    self._save_training_curves = save_training_curves
    self._sample_pretext_without_replacement = sample_pretext_without_replacement
    self._pretext_task = [benchmarker_wrapper().GetPretextTask() for
                           benchmarker_wrapper in benchmarker_wrappers]
    self._pretext_params = [benchmarker_wrapper().GetPretextParams() for
                           benchmarker_wrapper in benchmarker_wrappers]
    self._training_scheme = [benchmarker_wrapper().GetTrainingScheme() for
                             benchmarker_wrapper in benchmarker_wrappers]

  def process(self, element):
    output_data = {}
    output_data.update(element['generator_config'])
    output_data['marginal_params'] = element['marginal_param']
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
         pretext_task,
         pretext_params,
         training_scheme) in zip(self._benchmarker_classes,
                          self._benchmark_params,
                          self._model_classes,
                          self._h_params,
                          self._pretext_task,
                          self._pretext_params,
                          self._training_scheme):
      print(f'Running {model_class.__name__}_{pretext_task.__name__ if pretext_task is not None else "baseline"}_{training_scheme}')

      num_possible_configs = ComputeNumPossibleConfigs(benchmark_params, h_params, pretext_params)
      num_tuning_rounds = min(num_possible_configs, self._num_tuning_rounds)

      if num_tuning_rounds == 1 or self._tuning_metric == '':
        benchmark_params_sample, h_params_sample, pretext_params_sample = SampleModelConfig(benchmark_params,
                                                                     h_params, pretext_params)
        benchmarker = benchmarker_class(element['generator_config'],
                                        model_class,
                                        benchmark_params_sample,
                                        h_params_sample,
                                        pretext_task,
                                        pretext_params_sample,
                                        training_scheme)
        benchmarker_out = benchmarker.Benchmark(element,
                                                tuning_metric=self._tuning_metric,
                                                tuning_metric_is_loss=self._tuning_metric_is_loss)
        val_metrics = benchmarker_out['val_metrics']
        test_metrics = benchmarker_out['test_metrics']
        pretext_losses = benchmarker_out['pretext_losses']
        downstream_train_losses = benchmarker_out['downstream_train_losses']
        downstream_val_losses = benchmarker_out['downstream_val_losses']
        downstream_val_tuning_metrics = benchmarker_out['downstream_val_tuning_metrics']
        skipped = benchmarker_out['skipped']

      else:
        configs = []
        val_metrics_list = []
        test_metrics_list = []
        pretext_losses_list = []
        downstream_val_losses_list = []
        downstream_train_losses_list = []
        downstream_val_tuning_metrics_list = []
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
          if pretext_params is None:
            pretext_params_product = []
          else:
            pretext_params_product = list(GetCartesianProduct(pretext_params))
          num_pretext_configs = len(pretext_params_product)
          if num_pretext_configs > 0:
            num_tuning_rounds *= num_pretext_configs
          full_product = True
        elif self._sample_pretext_without_replacement:
          benchmark_params_sample, h_params_sample, _ = SampleModelConfig(benchmark_params,
                                                                         h_params, pretext_params)
          pretext_params_product = list(GetCartesianProduct(pretext_params))
          num_tuning_rounds = len(pretext_params_product)
          random.shuffle(pretext_params_product)
        for i in range(num_tuning_rounds):
          print(i)
          if full_product:
            if num_benchmark_configs > 0:
              benchmark_index = math.floor(i / (num_h_configs*num_pretext_configs))
              benchmark_params_sample = benchmark_params_product[benchmark_index]
            else:
              benchmark_params_sample = None
            if num_h_configs > 0:
              h_index = math.floor(i / num_pretext_configs)
              h_params_sample = h_params_product[h_index]
            else:
              h_params_sample = None
            if num_pretext_configs > 0:
              p_index = i % num_pretext_configs
              pretext_params_sample = pretext_params_product[p_index]
            else:
              pretext_params_sample = None
          elif self._sample_pretext_without_replacement:
            pretext_params_sample = pretext_params_product[i]
          else:
            benchmark_params_sample, h_params_sample, pretext_params_sample = SampleModelConfig(benchmark_params,
                                                                         h_params, pretext_params)
          benchmarker = benchmarker_class(element['generator_config'],
                                          model_class,
                                          benchmark_params_sample,
                                          h_params_sample,
                                          pretext_task,
                                          pretext_params_sample,
                                          training_scheme)
          benchmarker_out = benchmarker.Benchmark(element,
                                                  tuning_metric=self._tuning_metric,
                                                  tuning_metric_is_loss=self._tuning_metric_is_loss)

          if not benchmarker_out['skipped']:
            configs.append((benchmark_params_sample, h_params_sample, pretext_params_sample))
            val_metrics_list.append(benchmarker_out['val_metrics'])
            test_metrics_list.append(benchmarker_out['test_metrics'])
            pretext_losses_list.append(benchmarker_out['pretext_losses'])
            downstream_train_losses_list.append(benchmarker_out['downstream_train_losses'])
            downstream_val_losses_list.append(benchmarker_out['downstream_val_losses'])
            downstream_val_tuning_metrics_list.append(benchmarker_out['downstream_val_tuning_metrics'])

          if benchmarker_out['val_metrics'][self._tuning_metric] == 1.0 and not self._tuning_metric_is_loss:
            print("stopped")
            break # stop tuning if perfect evaluation metric has been achieved

        val_scores = [metrics[self._tuning_metric] for metrics in val_metrics_list]
        test_scores = [metrics[self._tuning_metric] for metrics in test_metrics_list]
        if self._tuning_metric_is_loss:
          best_tuning_round = np.argmin(val_scores)
        else:
          best_tuning_round = np.argmax(val_scores)
        benchmark_params_sample, h_params_sample, pretext_params_sample = configs[best_tuning_round]

        output_data['%s_%s_%s_num_tuning_rounds' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = num_tuning_rounds
        if self._save_tuning_results:
          output_data['%s_%s_%s_configs' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = configs
          output_data['%s_%s_%s_val_scores' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = val_scores
          output_data['%s_%s_%s_test_scores' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = test_scores
          output_data['%s_%s_%s_pretext_losses' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = pretext_losses_list
          output_data['%s_%s_%s_downstream_val_losses' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = downstream_val_losses_list
          output_data['%s_%s_%s_downstream_train_losses' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = downstream_train_losses_list
          output_data['%s_%s_%s_downstream_val_tuning_metrics' % (benchmarker.GetModelName(), benchmarker.GetPretextTaskName(), training_scheme)] = downstream_val_tuning_metrics_list


        val_metrics = val_metrics_list[best_tuning_round]
        test_metrics = test_metrics_list[best_tuning_round]
        pretext_losses = pretext_losses_list[best_tuning_round]
        downstream_val_losses = downstream_val_losses_list[best_tuning_round]
        downstream_train_losses = downstream_train_losses_list[best_tuning_round]
        downstream_val_tuning_metrics = downstream_val_tuning_metrics_list[best_tuning_round]
        skipped = False # Hack as this will be skipped if all tuning rounds fails. TODO fix this

      # Return benchmark data for next beam stage.

      for key, value in val_metrics.items():
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_val_{key}'] = value
      for key, value in test_metrics.items():
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_test_{key}'] = value


      if benchmark_params_sample is not None:
        for key, value in benchmark_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_train_{key}'] = value

      if h_params_sample is not None:
        for key, value in h_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_encoder_{key}'] = value

      if pretext_params_sample is not None:
        for key, value in pretext_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_pretext_{key}'] = value

      output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_skipped'] = skipped
      if self._save_training_curves:
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_pretext_losses'] = pretext_losses
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_downstream_val_losses'] = downstream_val_losses
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_downstream_train_losses'] = downstream_train_losses
        output_data[f'{benchmarker.GetModelName()}_{benchmarker.GetPretextTaskName()}_{training_scheme}_downstream_val_tuning_metrics'] = downstream_val_tuning_metrics

    yield json.dumps(output_data)





@gin.configurable
class NodeClassificationBeamHandlerSSL(NodeClassificationBeamHandler):

  @gin.configurable
  def __init__(self, benchmarker_wrappers, generator_wrapper,
               num_tuning_rounds=1, tuning_metric='',
               tuning_metric_is_loss=False, ktrain=5, ktuning=5,
               save_tuning_results=False, save_training_curves=False, sample_pretext_without_replacement = False):
    super().__init__(benchmarker_wrappers, generator_wrapper,
               num_tuning_rounds=num_tuning_rounds, tuning_metric=tuning_metric,
               tuning_metric_is_loss=tuning_metric_is_loss, ktrain=ktrain, ktuning=ktuning,
               save_tuning_results=save_tuning_results)

    self._benchmark_par_do = BenchmarkGNNParDoSSL(
        benchmarker_wrappers, num_tuning_rounds, tuning_metric,
        tuning_metric_is_loss, save_tuning_results, save_training_curves, sample_pretext_without_replacement)