# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for the PNASNet classification networks.

Paper: https://arxiv.org/abs/1712.00559
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from cell import PNASCell

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim


# Notes for training large PNASNet model on ImageNet
# -------------------------------------
# batch size (per replica): 16
# learning rate: 0.015 * 100
# learning rate decay factor: 0.97
# num epochs per decay: 2.4
# sync sgd with 100 replicas
# auxiliary head loss weighting: 0.4
# label smoothing: 0.1
# clip global norm of all gradients by 10
def large_imagenet_config():
  """Large ImageNet configuration based on PNASNet-5."""
  return tf.contrib.training.HParams(
      stem_multiplier=3.0,
      dense_dropout_keep_prob=0.5,
      num_cells=12,
      filter_scaling_rate=2.0,
      num_conv_filters=216,
      drop_path_keep_prob=0.6,
      use_aux_head=1,
      num_reduction_layers=2,
      total_training_steps=250000,
      num_stem_cells=2,
  )


def calc_reduction_layers(num_cells, num_reduction_layers):
  """Figure out what layers should have reductions."""
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers


def pnasnet_large_arg_scope(weight_decay=4e-5,
                            batch_norm_decay=0.9997,
                            batch_norm_epsilon=1e-3):
  batch_norm_params = {
      'decay': batch_norm_decay, # decay for the moving averages
      'epsilon': batch_norm_epsilon, # epsilon to prevent 0s in variance
      'scale': True,
      'fused': True,
  }
  weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      mode='FAN_OUT')
  with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d],
                 weights_regularizer=weights_regularizer,
                 weights_initializer=weights_initializer):
    with arg_scope([slim.fully_connected],
                   activation_fn=None, scope='FC'):
      with arg_scope([slim.conv2d, slim.separable_conv2d],
                     activation_fn=None, biases_initializer=None):
        with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
          return sc


def _imagenet_stem(inputs, hparams, stem_cell):
  """Stem used for models trained on ImageNet."""
  num_stem_filters = int(32 * hparams.stem_multiplier)
  net = slim.conv2d(inputs, num_stem_filters, [3, 3], stride=2, scope='conv0',
      padding='VALID')
  net = slim.batch_norm(net, scope='conv0_bn')

  # Run the reduction cells
  cell_outputs = [None, net]
  filter_scaling = 1.0 / (hparams.filter_scaling_rate**hparams.num_stem_cells)
  for cell_num in range(hparams.num_stem_cells):
    net = stem_cell(
        net,
        scope='cell_stem_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=2,
        prev_layer=cell_outputs[-2],
        cell_num=cell_num)
    cell_outputs.append(net)
    filter_scaling *= hparams.filter_scaling_rate
  return net, cell_outputs


def _build_aux_head(net, end_points, num_classes, hparams, scope):
  """Auxiliary head used for all models across all datasets."""
  with tf.variable_scope(scope):
    aux_logits = tf.identity(net)
    with tf.variable_scope('aux_logits'):
      aux_logits = slim.avg_pool2d(
          aux_logits, [5, 5], stride=3, padding='VALID')
      aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
      aux_logits = slim.batch_norm(aux_logits, scope='aux_bn0')
      aux_logits = tf.nn.relu(aux_logits)
      # Shape of feature map before the final layer.
      shape = aux_logits.shape
      shape = shape[1:3]
      aux_logits = slim.conv2d(aux_logits, 768, shape, padding='VALID')
      aux_logits = slim.batch_norm(aux_logits, scope='aux_bn1')
      aux_logits = tf.nn.relu(aux_logits)
      aux_logits = tf.contrib.layers.flatten(aux_logits)
      aux_logits = slim.fully_connected(aux_logits, num_classes)
      end_points['AuxLogits'] = aux_logits


def build_pnasnet_large(images,
                        num_classes,
                        is_training=True,
                        final_endpoint=None,
                        config=None):
  """Build PNASNet Large model for the ImageNet Dataset."""
  hparams = large_imagenet_config()
  if not is_training:
    hparams.set_hparam('drop_path_keep_prob', 1.0)

  total_num_cells = hparams.num_cells + hparams.num_stem_cells
  cell = PNASCell(hparams.num_conv_filters, hparams.drop_path_keep_prob,
                  total_num_cells, hparams.total_training_steps)

  with arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):
    end_points = {}
    def add_and_check_endpoint(endpoint_name, net):
      end_points[endpoint_name] = net
      return final_endpoint and (endpoint_name == final_endpoint)

    # Find where to place the reduction cells or stride normal cells
    reduction_indices = calc_reduction_layers(
        hparams.num_cells, hparams.num_reduction_layers)

    net, cell_outputs = _imagenet_stem(images, hparams, cell)
    if add_and_check_endpoint('Stem', net):
      return net, end_points

    # Setup for building in the auxiliary head.
    aux_head_cell_idxes = []
    if len(reduction_indices) >= 2:
      aux_head_cell_idxes.append(reduction_indices[1] - 1)

    # Run the cells
    filter_scaling = 1.0
    for cell_num in range(hparams.num_cells):
      is_reduction = cell_num in reduction_indices
      stride = 2 if is_reduction else 1
      if is_reduction: filter_scaling *= hparams.filter_scaling_rate
      net = cell(
          net,
          scope='cell_{}'.format(cell_num),
          filter_scaling=filter_scaling,
          stride=stride,
          prev_layer=cell_outputs[-2],
          cell_num=hparams.num_stem_cells + cell_num)
      if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
        return net, end_points
      cell_outputs.append(net)

      if (hparams.use_aux_head and cell_num in aux_head_cell_idxes and
          num_classes and is_training):
        aux_net = tf.nn.relu(net)
        _build_aux_head(aux_net, end_points, num_classes, hparams,
                        scope='aux_{}'.format(cell_num))

    # Final softmax layer
    with tf.variable_scope('final_layer'):
      net = tf.nn.relu(net)
      net = tf.reduce_mean(net, [1, 2])
      if add_and_check_endpoint('global_pool', net) or not num_classes:
        return net, end_points

      net = slim.dropout(net, hparams.dense_dropout_keep_prob, scope='dropout')
      logits = slim.fully_connected(net, num_classes)
      if add_and_check_endpoint('Logits', logits):
        return net, end_points

      predictions = tf.nn.softmax(logits, name='predictions')
      if add_and_check_endpoint('Predictions', predictions):
        return net, end_points

    return logits, end_points
