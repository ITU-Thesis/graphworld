# Copyright 2022 Google LLC
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
"""
There are currently 3 pieces required for each model:

  * BenchmarkerWrapper (ex. NodeGCN) -- Used in GIN config, this delegates to the Benchmarker.
  * ModelBenchmarker (ex. GCNNodeBenchmarker) -- This performs the actual training and eval steps for the model
  * Modelmpl (ex. GCNNodeModel) -- This is the actual model implemention (wrapping together convolution layers)
"""
import copy
import gin
import numpy as np
import sklearn.metrics
import torch
from torch.nn import Linear
import copy
from graph_world.models.basic_gnn import BasicGNN
from graph_world.self_supervised_learning.pretext_tasks.basic_pretext_task import BasicPretextTask, IdentityPretextTask
from typing import Type, List
import inspect
import logging


from ..beam.benchmarker import BenchmarkerWrapper
from ..nodeclassification.benchmarker import NNNodeBenchmarker
from  . import *
from .pretext_tasks.__types import *

    
class NNNodeBenchmarkerJL(NNNodeBenchmarker):
  def __init__(self, generator_config : dict, model_class : BasicGNN, benchmark_params : dict, h_params : dict, 
               pretext_task : BasicPretextTask):
    super(NNNodeBenchmarker, self).__init__(generator_config, model_class, benchmark_params, h_params)
    self._epochs = benchmark_params['epochs']
    self._lr = benchmark_params['lr']
    self.downstream_out = h_params['out_channels']

    # pretext_tasks, names and weights
    self._pretext_task = pretext_task if pretext_task is not None else IdentityPretextTask
    self._lambda = benchmark_params['lambda'] # Weight for summing loss of pretext task
    self._pretext_task_name = pretext_task.__name__ if pretext_task is not None else ""

    # Encoder end pretext models both have hparams in the same dict
    # The available graph encoders cannot handle unknown params, so we
    # need to remove them. 
    # We copy since tuning and logging neeeds access to the original values
    self._pretext_h_params = copy.deepcopy(h_params)
    self._encoder_h_params = copy.deepcopy(h_params)

    # Remove pretext hparams from encoder params
    parameters = inspect.signature(pretext_task.__init__).parameters 
    for k,v in parameters.items():
        if k not in ['self', 'args', 'kwargs']:
          self._encoder_h_params.pop(k, None) # delete pretext hparams from encoder

    # Modify encoder output dim and instantiate encoder
    self._encoder_h_params['out_channels'] = self._encoder_h_params['hidden_channels']
    self._encoder = model_class(**self._encoder_h_params)
    
    # Give encoder instantiation to pretext tasks, 
    # so they can run modified inputs through the encoder
    self._pretext_h_params['encoder'] = self._encoder
    
    # Give number of epochs the pretext task will run for
    # For joint learning this is the same as epochs
    self._pretext_h_params['epochs'] = self._epochs

    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None

  def GetPretextTaskName(self):
    return self._pretext_task_name


  def train_step(self, data : InputGraph):
    self._encoder.train()
    self._downstream_decoder.train()
    self._pretext_model.decoder.train()
    self._optimizer.zero_grad()  # Clear gradients.

    # Compute downstream loss
    embeddings = self._pretext_model.get_downstream_embeddings() 
    downstream_out = self._downstream_decoder(embeddings) # downstream predictions
    loss = self._criterion(downstream_out[self._train_mask],
                           data.y[self._train_mask])

    # Add pretext loss
    loss += self._lambda * self._pretext_model.make_loss(embeddings)
    
    # Update parameters
    loss.backward()
    self._optimizer.step()
    return loss


  def test(self, data : InputGraph, test_on_val : bool = False) -> EvaluationMetrics:
    self._encoder.eval()
    self._downstream_decoder.eval()
    self._pretext_model.decoder.eval()

    embeddings = self._pretext_model.get_downstream_embeddings()
    out = self._downstream_decoder(embeddings)
    if test_on_val:
      pred = out[self._val_mask].detach().numpy()
    else:
      pred = out[self._test_mask].detach().numpy()

    pred_best = pred.argmax(-1)
    if test_on_val:
      correct = data.y[self._val_mask].numpy()
    else:
      correct = data.y[self._test_mask].numpy()
    n_classes = out.shape[-1]
    pred_onehot = np.zeros((len(pred_best), n_classes))
    pred_onehot[np.arange(pred_best.shape[0]), pred_best] = 1

    correct_onehot = np.zeros((len(correct), n_classes))
    correct_onehot[np.arange(correct.shape[0]), correct] = 1

    results : EvaluationMetrics = {
        'accuracy': sklearn.metrics.accuracy_score(correct, pred_best),
        'f1_micro': sklearn.metrics.f1_score(correct, pred_best,
                                                  average='micro'),
        'f1_macro': sklearn.metrics.f1_score(correct, pred_best,
                                                  average='macro'),
        'rocauc_ovr': sklearn.metrics.roc_auc_score(correct_onehot,
                                                         pred_onehot,
                                                         multi_class='ovr'),
        'rocauc_ovo': sklearn.metrics.roc_auc_score(correct_onehot,
                                                         pred_onehot,
                                                         multi_class='ovo'),
        'logloss': sklearn.metrics.log_loss(correct, pred)
    }
    return results


  def train(self, data : InputGraph, tuning_metric: str, tuning_metric_is_loss: bool):

    # Setup pretext task
    self._pretext_h_params['data'] = data
    self._pretext_h_params['train_mask'] = self._train_mask
    self._pretext_model = self._pretext_task(**self._pretext_h_params) # init pretext with hparams
    
    # setup downstream decoder
    self._downstream_decoder = Linear(self._pretext_model.get_embedding_dim(), self.downstream_out)

    # Setup optimizer
    params = list(self._encoder.parameters())
    params += list(self._downstream_decoder.parameters())
    params += list(self._pretext_model.decoder.parameters())
    self._optimizer = torch.optim.Adam(params,
                                    lr=self._lr,
                                    weight_decay=5e-4)

    # Train for x epochs
    losses = []
    tuning_metrics = []
    best_val_metric = np.inf if tuning_metric_is_loss else -np.inf
    test_metrics = None
    best_val_metrics = None
    for _ in range(self._epochs):
      losses.append(float(self.train_step(data)))
      val_metrics = self.test(data, test_on_val=True)
      tuning_metrics.append(val_metrics[tuning_metric])
      if ((tuning_metric_is_loss and val_metrics[tuning_metric] < best_val_metric) or
          (not tuning_metric_is_loss and val_metrics[tuning_metric] > best_val_metric)):
        best_val_metric = val_metrics[tuning_metric]
        best_val_metrics = copy.deepcopy(val_metrics)
        test_metrics = self.test(data, test_on_val=False)
    return losses, tuning_metrics, test_metrics, best_val_metrics


  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
      'skipped': skipped,
      'results': None
    }
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {}
    out['test_metrics'] = {}

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    train_mask, val_mask, test_mask = masks

    self.SetMasks(train_mask, val_mask, test_mask)

    val_metrics = {}
    test_metrics = {}
    losses = None
    try:
      losses, tuning_metrics, test_metrics, val_metrics = self.train(
        torch_data, tuning_metric=tuning_metric, tuning_metric_is_loss=tuning_metric_is_loss)
    except Exception as e:
      raise e
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    out['losses'] = losses
    out['tuning_metrics'] = tuning_metrics
    out['test_metrics'].update(test_metrics)
    out['val_metrics'].update(val_metrics)
    return out


@gin.configurable
class NNNodeBenchmarkJL(BenchmarkerWrapper):
  def __init__(self, model_class : BasicGNN = None, benchmark_params : dict = None, h_params : dict = None, 
    pretext_task : BasicPretextTask = None):
    super().__init__(model_class, benchmark_params, h_params)
    self._pretext_task = pretext_task

  def GetBenchmarker(self) -> NNNodeBenchmarkerJL:
    return NNNodeBenchmarkerJL(self._model_class, self._benchmark_params, self._h_params, self._pretext_task)

  def GetBenchmarkerClass(self) -> Type[NNNodeBenchmarkerJL]:
    return NNNodeBenchmarkerJL

  def GetPretextTask(self) -> BasicPretextTask:
    return self._pretext_task
