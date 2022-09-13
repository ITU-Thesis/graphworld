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
import logging
import numpy as np
import graph_tool.all as gt
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import torch
from torch_geometric.nn import GAE

from .utils import MseWrapper
from .models import PyGBasicGraphModel
from .benchmarker import Benchmarker, BenchmarkerWrapper


@gin.configurable
class NNGraphBenchmark(BenchmarkerWrapper):

  def __init__(self, model_class, benchmark_params, h_params):
    self._model_class = model_class
    self._benchmark_params = benchmark_params
    self._h_params = h_params

  def GetBenchmarker(self):
    return NNGraphBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return NNGraphBenchmarker


class NNGraphBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    self._epochs = self._benchmark_params['epochs']
    self._lr = self._benchmark_params['lr']
    self._model = PyGBasicGraphModel(self._model_class, self._h_params)
    # TODO(palowitch) make optimizer configurable
    self._optimizer = torch.optim.Adam(self._model.parameters(), self._lr, weight_decay=5e-4)
    self._criterion = torch.nn.MSELoss()

  def train(self, loader):
    self._model.train()
    train_losses = []
    train_mse = []
    for epoch in range(1, self._epochs):
      for iter, data in enumerate(loader):  # Iterate in batches over the training dataset.
        try:
          out = self._model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        except IndexError:
          print(iter)
          print(data)
          raise
        loss = self._criterion(out[:, 0], data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        self._optimizer.step()  # Update parameters based on gradients.
        self._optimizer.zero_grad()  # Clear gradients.
      train_mse.append(float(self.test(loader)[0]))
      train_losses.append(float(loss))
    return train_mse, train_losses

  def test(self, loader):
    self._model.eval()
    predictions = []
    labels = []
    for iter, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
      batch_size = data.batch.size().numel()
      out = self._model(data.x, data.edge_index, data.batch)
      predictions.append(out[:, 0].cpu().detach().numpy())
      labels.append(data.y.cpu().detach().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    mse = MseWrapper(predictions, labels)
    mse_scaled = MseWrapper(predictions, labels, scale=True)
    return mse, mse_scaled

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    sample_id = element['sample_id']
    val_mse = 0.0
    val_mse_scaled = 0.0
    test_mse = 0.0
    test_mse_scaled = 0.0
    try:
      mses, losses = self.train(element['torch_dataset']['train'])
      total_mse_val, total_mse_scaled_val = self.test(element['torch_dataset']['tuning'])
      total_mse_test, total_mse_scaled_test = self.test(element['torch_dataset']['test'])
      val_mse = float(total_mse_val)
      val_mse_scaled = float(total_mse_scaled_val)
      test_mse = float(total_mse_test)
      test_mse_scaled = float(total_mse_scaled_test)
    except Exception:
      logging.info(f'Failed to run for sample id {sample_id}')
      losses = None

    return {'losses': losses,
            'val_metrics': {'mse': val_mse,
                            'mse_scaled': val_mse_scaled},
            'test_metrics': {'mse': test_mse,
                             'mse_scaled': test_mse_scaled}}


class LRGraphBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class=None,
               benchmark_params=None, h_params=None):
    super().__init__(generator_config, model_class, benchmark_params,
                     h_params)
    self._model_name = 'LR'

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    reg = LinearRegression().fit(
      element['numpy_dataset']['train']['X'],
      element['numpy_dataset']['train']['y'])
    y_pred_val = reg.predict(element['numpy_dataset']['tuning']['X'])
    y_pred_test = reg.predict(element['numpy_dataset']['test']['X'])
    y_val = element['numpy_dataset']['tuning']['y']
    y_test = element['numpy_dataset']['test']['y']
    val_mse = MseWrapper(y_pred_val, y_val)
    val_mse_scaled = MseWrapper(y_pred_val, y_val, scale=True)
    test_mse = MseWrapper(y_pred_test, y_test)
    test_mse_scaled = MseWrapper(y_pred_test, y_test, scale=True)
    return {'losses': [],
            'val_metrics': {'mse': val_mse,
                            'mse_scaled': val_mse_scaled},
            'test_metrics': {'mse': test_mse,
                             'mse_scaled': test_mse_scaled}}

@gin.configurable
class LRGraphBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LRGraphBenchmarker(**self._h_params)

  def GetBenchmarkerClass(self):
    return LRGraphBenchmarker


# general benchmarkers
class NNNodeBenchmarker(Benchmarker):
  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']

    self._model = model_class(**h_params)
    # TODO(palowitch): make optimizer configurable.
    self._optimizer = torch.optim.Adam(self._model.parameters(),
                                       lr=benchmark_params['lr'],
                                       weight_decay=5e-4)
    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None

  def AdjustParams(self, generator_config):
    if 'num_clusters' in generator_config and self._h_params is not None:
      self._h_params['out_channels'] = generator_config['num_clusters']

  def SetMasks(self, train_mask, val_mask, test_mask):
    self._train_mask = train_mask
    self._val_mask = val_mask
    self._test_mask = test_mask

  def train_step(self, data):
    self._model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    out = self._model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = self._criterion(out[self._train_mask],
                           data.y[self._train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data, test_on_val=False):
    self._model.eval()
    out = self._model(data.x, data.edge_index)
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

    results = {
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
        'logloss': sklearn.metrics.log_loss(correct, pred)}
    return results

  def train(self, data,
            tuning_metric: str,
            tuning_metric_is_loss: bool):
    losses = []
    best_val_metric = np.inf if tuning_metric_is_loss else -np.inf
    test_metrics = None
    best_val_metrics = None
    for i in range(self._epochs):
      losses.append(float(self.train_step(data)))
      val_metrics = self.test(data, test_on_val=True)
      if ((tuning_metric_is_loss and val_metrics[tuning_metric] < best_val_metric) or
          (not tuning_metric_is_loss and val_metrics[tuning_metric] > best_val_metric)):
        best_val_metric = val_metrics[tuning_metric]
        best_val_metrics = copy.deepcopy(val_metrics)
        test_metrics = self.test(data, test_on_val=False)
    return losses, test_metrics, best_val_metrics

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
      losses, test_metrics, val_metrics = self.train(
        torch_data, tuning_metric=tuning_metric, tuning_metric_is_loss=tuning_metric_is_loss)
    except Exception as e:
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    out['losses'] = losses
    out['test_metrics'].update(test_metrics)
    out['val_metrics'].update(val_metrics)
    return out


class NNNodeBaselineBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    # remove meta entries from h_params
    self._alpha = h_params['alpha']

  def GetModelName(self):
    return 'PPRBaseline'

  def test(self, data, graph, masks, test_on_val=False):
    train_mask, val_mask, test_mask = masks
    node_ids = np.arange(train_mask.shape[0])
    labels = data.y.numpy()
    nodes_train, nodes_val, nodes_test = node_ids[train_mask], node_ids[val_mask], node_ids[test_mask]
    n_classes = max(data.y.numpy()) + 1
    pers = graph.new_vertex_property("double")
    if test_on_val:
      pred = np.zeros((len(nodes_val), n_classes))
      for idx, node in enumerate(nodes_val):
        pers.a = 0
        pers[node] = 1
        pprs = np.array(gt.pagerank(graph, damping=1-self._alpha, pers=pers, max_iter=100).a)
        pred[idx, labels[nodes_train]] += pprs[nodes_train]
    else:
      pred = np.zeros((len(nodes_test), n_classes))
      for idx, node in enumerate(nodes_test):
        pers.a = 0
        pers[node] = 1
        pprs = np.array(gt.pagerank(graph, damping=1-self._alpha, pers=pers, max_iter=100).a)
        pred[idx, labels[nodes_train]] += pprs[nodes_train]

    pred_best = pred.argmax(-1)
    if test_on_val:
      correct = labels[nodes_val]
    else:
      correct = labels[nodes_test]

    pred_onehot = np.zeros((len(pred_best), n_classes))
    pred_onehot[np.arange(pred_best.shape[0]), pred_best] = 1

    correct_onehot = np.zeros((len(correct), n_classes))
    correct_onehot[np.arange(correct.shape[0]), correct] = 1

    results = {
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
        'logloss': sklearn.metrics.log_loss(correct, pred)}
    return results

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    gt_data = element['gt_data']
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

    try:
      # Divide by zero sometimes happens with the ksample masks.
      out['val_metrics'].update(self.test(torch_data, gt_data, masks, test_on_val=True))
      out['test_metrics'].update(self.test(torch_data, gt_data, masks, test_on_val=False))
    except Exception:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')
      out['skipped'] = True

    return out

@gin.configurable
class NNNodeBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return NNNodeBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return NNNodeBenchmarker

@gin.configurable
class NNNodeBaselineBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return NNNodeBaselineBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return NNNodeBaselineBenchmarker


# Link prediction
class LPBenchmarker(Benchmarker):
  def __init__(self, generator_config, model_class, benchmark_params, h_params):

    super().__init__(generator_config, model_class, benchmark_params, h_params)

    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']
    self._model = self._model_class(**h_params)
    self._lp_wrapper_model = GAE(self._model)
    # TODO(palowitch,tsitsulin): fill optimizer using param input instead.
    self._optimizer = torch.optim.Adam(self._model.parameters(),
                                       lr=benchmark_params['lr'],
                                       weight_decay=5e-4)

  def AdjustParams(self, generator_config):
    if 'num_clusters' in generator_config:
      self._h_params['out_channels'] = generator_config['num_clusters']

  def train_step(self, data):
    self._model.train()
    self._lp_wrapper_model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    z = self._model(data.x, data.train_pos_edge_index)
    loss = self._lp_wrapper_model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data, test_on_val=False):
    self._model.eval()
    self._lp_wrapper_model.eval()
    results = {}
    z = self._model(data.x, data.train_pos_edge_index)

    if test_on_val:
      roc_auc_score, average_precision_score = self._lp_wrapper_model.test(
        z, data.val_pos_edge_index, data.val_neg_edge_index)
    else:
      roc_auc_score, average_precision_score = self._lp_wrapper_model.test(
        z, data.test_pos_edge_index, data.test_neg_edge_index)
    results['rocauc'] = roc_auc_score
    results['ap'] = average_precision_score
    return results

  def train(self, data):
    losses = []
    for epoch in range(self._epochs):
      losses.append(float(self.train_step(data)))
    return losses

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    torch_data = element['torch_data']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
        'skipped': skipped,
        'results': None
    }
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {
        'rocauc': 0,
        'ap': 0,
    }
    out['test_metrics'] = {
        'rocaus': 0,
        'ap': 0,
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    out['losses'] = None
    try:
      out['losses'] = self.train(torch_data)
      # Divide by zero sometimes happens with the ksample masks.
      out['val_metrics'].update(self.test(torch_data, test_on_val=True))
      out['test_metrics'].update(self.test(torch_data, test_on_val=False))
    except Exception:
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    return out


class LPBaselineBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    self._scorer = h_params['scorer']

  def score(self, graph, edge_index):
    return gt.vertex_similarity(graph, sim_type=self._scorer,
                                vertex_pairs=edge_index)

  def test(self, data, test_on_val=False):
    graph = gt.Graph(directed=False)
    graph.add_vertex(data.y.shape[0])
    graph.add_edge_list(data.train_pos_edge_index.T)

    if test_on_val:
      pos_scores = self.score(graph, data.val_pos_edge_index.T)
      neg_scores = self.score(graph, data.val_neg_edge_index.T)
      y_true = np.ones(data.val_pos_edge_index.shape[1] +
                       data.val_neg_edge_index.shape[1])
      y_true[data.val_pos_edge_index.shape[1]:] = 0
    else:
      pos_scores = self.score(graph, data.test_pos_edge_index.T)
      neg_scores = self.score(graph, data.test_neg_edge_index.T)
      y_true = np.ones(data.test_pos_edge_index.shape[1] +
                       data.test_neg_edge_index.shape[1])
      y_true[data.test_pos_edge_index.shape[1]:] = 0

    all_scores = np.hstack([pos_scores, neg_scores])
    all_scores = np.nan_to_num(all_scores, copy=False)

    return {
        'rocauc': sklearn.metrics.roc_auc_score(y_true, all_scores),
        'ap': sklearn.metrics.average_precision_score(y_true, all_scores),
    }

  def GetModelName(self):
    return 'Baseline'

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    torch_data = element['torch_data']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
        'skipped': skipped,
        'results': None
    }
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {
        'rocauc': 0,
        'ap': 0,
    }
    out['test_metrics'] = {
        'rocauc': 0,
        'ap': 0,
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    try:
      # Divide by zero sometimes happens with the ksample masks.
      out['val_metrics'].update(self.test(torch_data, test_on_val=True))
      out['test_metrics'].update(self.test(torch_data, test_on_val=False))
    except Exception:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')
      out['skipped'] = True

    return out

@gin.configurable
class LPBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LPBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return LPBenchmarker

@gin.configurable
class LPBenchmarkBaseline(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LPBaselineBenchmarker(None, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return LPBaselineBenchmarker


class NodeRegressionBenchmarker(Benchmarker):
  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']

    self._model = model_class(**h_params)
    # TODO(palowitch): make optimizer configurable.
    self._optimizer = torch.optim.Adam(self._model.parameters(),
                                       lr=benchmark_params['lr'],
                                       weight_decay=5e-4)
    self._criterion = torch.nn.MSELoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None

  def SetMasks(self, train_mask, val_mask, test_mask):
    self._train_mask = train_mask
    self._val_mask = val_mask
    self._test_mask = test_mask

  def train_step(self, data):
    self._model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    out = self._model(data.x, data.edge_index).ravel()  # Perform a single forward pass.
    loss = self._criterion(out[self._train_mask],
                           data.y[self._train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data, test_on_val=False):
    self._model.eval()
    out = self._model(data.x, data.edge_index).ravel()
    if test_on_val:
      pred = out[self._val_mask].detach().numpy()
    else:
      pred = out[self._test_mask].detach().numpy()

    if test_on_val:
      correct = data.y[self._val_mask].numpy()
    else:
      correct = data.y[self._test_mask].numpy()

    results = {
        'mse': float(sklearn.metrics.mean_squared_error(correct, pred)),
    }
    return results

  def train(self, data,
            tuning_metric: str,
            tuning_metric_is_loss: bool):
    losses = []
    best_val_metric = np.inf if tuning_metric_is_loss else -np.inf
    best_val_metrics = None
    test_metrics = None
    for i in range(self._epochs):
      losses.append(float(self.train_step(data)))
      val_metrics = self.test(data, test_on_val=True)
      if ((tuning_metric_is_loss and val_metrics[tuning_metric] < best_val_metric) or
          (not tuning_metric_is_loss and val_metrics[tuning_metric] > best_val_metric)):
        best_val_metric = val_metrics[tuning_metric]
        best_val_metrics = copy.deepcopy(val_metrics)
        test_metrics = self.test(data, test_on_val=False)
    return losses, test_metrics, best_val_metrics


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
    out['val_metrics'] = {
        'mse': -1,
    }
    out['test_metrics'] = {
        'mse': -1,
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    train_mask, val_mask, test_mask = masks

    self.SetMasks(train_mask, val_mask, test_mask)

    out['losses'] = None
    try:
      losses, test_metrics, val_metrics = self.train(
        torch_data, tuning_metric=tuning_metric,
        tuning_metric_is_loss=tuning_metric_is_loss)
      out['losses'] = losses
      out['val_metrics'].update(val_metrics)
      out['test_metrics'].update(test_metrics)
    except Exception as e:
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    return out


@gin.configurable
class NodeRegressionBenchmark(BenchmarkerWrapper):
  def GetBenchmarker(self):
    return NodeRegressionBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return NodeRegressionBenchmarker
