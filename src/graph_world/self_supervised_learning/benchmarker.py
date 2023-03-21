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

    
class NNNodeBenchmarkerSSL(NNNodeBenchmarker):
  def __init__(self, generator_config : dict, model_class : BasicGNN, benchmark_params : dict, h_params : dict, 
               pretext_task : BasicPretextTask, pretext_params : dict, training_scheme : str):
    super(NNNodeBenchmarker, self).__init__(generator_config, model_class, benchmark_params, h_params)
    self._training_scheme = training_scheme
    # Set number of classes
    self._downstream_out = h_params['out_channels']

    # pretext_task and name. Set JL training scheme if no pretext task
    if pretext_task is None:
      self._pretext_task = IdentityPretextTask
      self._training_scheme = 'JL'
      self._pretext_task_name = ""
    else:
      self._pretext_task = pretext_task
      self._pretext_task_name = pretext_task.__name__

    # Set hparams. Deepcopy because we make mutations which are not needed for logging
    self._pretext_h_params = copy.deepcopy(pretext_params)
    self._encoder_h_params = copy.deepcopy(h_params)

    # Modify encoder output dim and instantiate encoder
    self._encoder_h_params['out_channels'] = self._encoder_h_params['hidden_channels']
    self._encoder = model_class(**self._encoder_h_params)
    
    # Give encoder instantiation to pretext task, 
    # so they can run modified inputs through the encoder
    self._pretext_h_params['encoder'] = self._encoder

    # Params for training scheme
    self._downstream_lr = benchmark_params['downstream_lr']
    self._downstream_epochs = benchmark_params['downstream_epochs']
    self._patience = benchmark_params['patience']
    self._pretext_h_params['pretext_weight'] = benchmark_params.get('pretext_weight', 1)
    if training_scheme in ['URL', 'PF']:
      self._pretext_epochs = benchmark_params['pretext_epochs']
      self._pretext_lr = benchmark_params['pretext_lr']
      self._pretext_h_params['epochs'] = self._pretext_epochs # set expected epochs for pretext
    else:
      self._pretext_h_params['epochs'] = self._downstream_epochs # set expected epochs for pretext
    
    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None

  def GetPretextTaskName(self):
    return self._pretext_task_name
  

  def GetTrainingScheme(self):
    return self._training_scheme
  

  def pretext_train_step(self, data : InputGraph):
    self._pretext_model.train()
    self._pretext_optimizer.zero_grad()  

    # Compute pretext loss
    # pretext_embeddings = self._pretext_model.get_pretext_embeddings(downstream_embeddings = None) 
    # loss = self._pretext_model.compute_pretext_loss(pretext_embeddings)

    embeddings = self._pretext_model.get_downstream_embeddings() 
    loss = self._pretext_model.make_loss(embeddings)
    
    # Update parameters
    loss.backward()
    self._pretext_optimizer.step()
    return loss


  def downstream_train_step(self, data : InputGraph):
    # Set train/eval modes
    self._downstream_decoder.train()
    if self._training_scheme in ['JL', 'PF']:
      self._pretext_model.train()
    else:
      self._encoder.eval()

    # Clear gradients
    self._downstream_optimizer.zero_grad()  

    # Compute downstream loss
    embeddings = self._pretext_model.get_downstream_embeddings() 
    downstream_out = self._downstream_decoder(embeddings) # downstream predictions
    loss = self._criterion(downstream_out[self._train_mask],
                           data.y[self._train_mask])

    # Add pretext loss
    if self._training_scheme in ['JL']:
        loss += self._pretext_h_params['pretext_weight'] * self._pretext_model.make_loss(embeddings)
    
    # Update parameters
    loss.backward()
    self._downstream_optimizer.step()
    return loss


  def test(self, data : InputGraph, test_on_val : bool = False) -> EvaluationMetrics:
    self._downstream_decoder.eval()
    self._pretext_model.eval()

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
    
    # Setup downstream decoder
    self._downstream_decoder = Linear(self._pretext_model.get_embedding_dim(), self._downstream_out)

    # Pretrain if two-stage training scheme
    pretext_losses = []
    if self._training_scheme in ['PF', 'URL']:
        self._pretext_optimizer = torch.optim.Adam(self._pretext_model.parameters(),
                                    lr=self._pretext_lr,
                                    weight_decay=5e-4)
        for _ in range(self._pretext_epochs):
          pretext_losses.append(float(self.pretext_train_step(data)))
        self._pretext_optimizer.zero_grad()  

    # Setup downstream optimizer
    params = list(self._downstream_decoder.parameters())
    if self._training_scheme in ['PF']:
      params += list(self._encoder.parameters())
    elif self._training_scheme in ['JL']:
      params += list(self._pretext_model.parameters())
    self._downstream_optimizer = torch.optim.Adam(params,
                                    lr=self._downstream_lr,
                                    weight_decay=5e-4)

    # Train downstream task
    downstream_train_losses = []
    downstream_val_losses = []
    downstream_val_tuning_metrics = []
    best_val_metric = np.inf if tuning_metric_is_loss else -np.inf
    test_metrics = None
    best_val_metrics = None
    last_improvement = 0
    for _ in range(self._downstream_epochs):
      if last_improvement == self._patience:
        break
      downstream_train_losses.append(float(self.downstream_train_step(data)))
      val_metrics = self.test(data, test_on_val=True)
      downstream_val_tuning_metrics.append(val_metrics[tuning_metric])
      downstream_val_losses.append(val_metrics['logloss'])
      if ((tuning_metric_is_loss and val_metrics[tuning_metric] < best_val_metric) or
          (not tuning_metric_is_loss and val_metrics[tuning_metric] > best_val_metric)):
        last_improvement = 0
        best_val_metric = val_metrics[tuning_metric]
        best_val_metrics = copy.deepcopy(val_metrics)
        test_metrics = self.test(data, test_on_val=False)
      else:
        last_improvement += 1
    return pretext_losses, downstream_train_losses, downstream_val_losses, downstream_val_tuning_metrics, test_metrics, best_val_metrics


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
    pretext_losses = None
    downstream_train_losses = None
    downstream_val_losses = None
    downstream_val_tuning_metrics = None    
    try:
      pretext_losses, downstream_train_losses, downstream_val_losses, downstream_val_tuning_metrics, test_metrics, val_metrics = self.train(
        torch_data, tuning_metric=tuning_metric, tuning_metric_is_loss=tuning_metric_is_loss)
    except Exception as e:
      print("FAILED")
      print(e)
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    out['pretext_losses'] = pretext_losses
    out['downstream_train_losses'] = downstream_train_losses
    out['downstream_val_losses'] = downstream_val_losses
    out['downstream_val_tuning_metrics'] = downstream_val_tuning_metrics
    out['test_metrics'].update(test_metrics)
    out['val_metrics'].update(val_metrics)
    return out


@gin.configurable
class NNNodeBenchmarkSSL(BenchmarkerWrapper):
  def __init__(self, model_class : BasicGNN = None, benchmark_params : dict = None, h_params : dict = None, 
    pretext_task : BasicPretextTask = None, pretext_params : dict = None, training_scheme : str = ''):
    super().__init__(model_class, benchmark_params, h_params)
    assert training_scheme in ['JL', 'PF', 'URL']
    self._pretext_task = pretext_task
    self._pretext_params = pretext_params
    self._training_scheme = training_scheme

  def GetBenchmarker(self) -> NNNodeBenchmarkerSSL:
    return NNNodeBenchmarkerSSL(self._model_class, self._benchmark_params, self._h_params, 
                                self._pretext_task, self._pretext_params, self._training_scheme)

  def GetBenchmarkerClass(self) -> Type[NNNodeBenchmarkerSSL]:
    return NNNodeBenchmarkerSSL

  def GetPretextTask(self) -> BasicPretextTask:
    return self._pretext_task
  
  def GetTrainingScheme(self) -> str:
    return self._training_scheme
  
  def GetPretextParams(self) -> dict:
    return self._pretext_params
