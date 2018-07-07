from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


class NASBaseCell(object):
  """NASNet Cell class that is used as a 'layer' in image architectures.

  Args:
    num_conv_filters: The number of filters for each convolution operation.
    operations: List of operations that are performed in the NASNet Cell in
      order.
    used_hiddenstates: Binary array that signals if the hiddenstate was used
      within the cell. This is used to determine what outputs of the cell
      should be concatenated together.
    hiddenstate_indices: Determines what hiddenstates should be combined
      together with the specified operations to create the NASNet cell.
  """

  def __init__(self, num_conv_filters, operations, used_hiddenstates,
               hiddenstate_indices, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    assert len(hiddenstate_indices) == len(operations)
    assert len(operations) % 2 == 0
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps

  def __call__(self, net, scope, filter_scaling, stride, prev_layer, cell_num):
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)

    with tf.variable_scope(scope):
      net = self._cell_base(net, prev_layer)
      for i in range(int(len(self._operations) / 2)):
        with tf.variable_scope('comb_iter_{}'.format(i)):
          h1 = net[self._hiddenstate_indices[i * 2]]
          h2 = net[self._hiddenstate_indices[i * 2 + 1]]
          with tf.variable_scope('left'):
            h1 = self._apply_operation(h1, self._operations[i * 2], stride, 
                                       self._hiddenstate_indices[i * 2] < 2)
          with tf.variable_scope('right'):
            h2 = self._apply_operation(h2, self._operations[i * 2 + 1], stride,
                                       self._hiddenstate_indices[i * 2 + 1] < 2)
          with tf.variable_scope('combine'):
            h = h1 + h2
          net.append(h)

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)

      return net

  def _cell_base(self, net, prev_layer):
    filter_size = self._filter_size

    if prev_layer is None:
      prev_layer = net
    elif net.shape[2] != prev_layer.shape[2]:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = self._factorized_reduction(prev_layer, filter_size, stride=2)
    elif filter_size != prev_layer.shape[3]:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = slim.conv2d(prev_layer, filter_size, 1, scope='prev_1x1')
      prev_layer = slim.batch_norm(prev_layer, scope='prev_bn')

    net = tf.nn.relu(net)
    net = slim.conv2d(net, filter_size, 1, scope='1x1')
    net = slim.batch_norm(net, scope='beginning_bn')
    net = tf.split(axis=3, num_or_size_splits=1, value=net)
    for split in net:
      assert split.shape[3] == filter_size
    net.append(prev_layer)
    return net

  def _apply_operation(self, net, operation, stride, is_from_original_input):
    if stride > 1 and not is_from_original_input:
      stride = 1
    input_filters = net.shape[3]
    filter_size = self._filter_size
    if 'separable' in operation:
      num_layers = int(operation.split('_')[-1])
      kernel_size = int(operation.split('x')[0][-1])
      for layer_num in range(num_layers):
        net = tf.nn.relu(net)
        net = slim.separable_conv2d(
            net,
            filter_size,
            kernel_size,
            depth_multiplier=1,
            scope='separable_{0}x{0}_{1}'.format(kernel_size, layer_num + 1),
            stride=stride)
        net = slim.batch_norm(
            net, scope='bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num + 1))
        stride = 1
    elif operation in ['none']:
      if stride > 1 or (input_filters != filter_size):
        net = tf.nn.relu(net)
        net = slim.conv2d(net, filter_size, 1, stride=stride, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    elif 'pool' in operation:
      pooling_type = operation.split('_')[0]
      pooling_shape = int(operation.split('_')[-1].split('x')[0])
      if pooling_type == 'avg':
        net = slim.avg_pool2d(net, pooling_shape, stride=stride, padding='SAME')
      elif pooling_type == 'max':
        net = slim.max_pool2d(net, pooling_shape, stride=stride, padding='SAME')
      else:
        raise ValueError('Unimplemented pooling type: ', pooling_type)
      if input_filters != filter_size:
        net = slim.conv2d(net, filter_size, 1, stride=1, scope='1x1')
        net = slim.batch_norm(net, scope='bn_1')
    else:
      raise ValueError('Unimplemented operation: ', operation)

    if operation != 'none':
      net = self._apply_drop_path(net)
    return net

  def _combine_unused_states(self, net):
    used_hiddenstates = self._used_hiddenstates
    states_to_combine = (
        [h for h, is_used in zip(net, used_hiddenstates) if not is_used])
    net = tf.concat(values=states_to_combine, axis=3)
    return net

  def _apply_drop_path(self, net):
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      # Scale keep prob by layer number
      assert self._cell_num != -1
      layer_ratio = (self._cell_num + 1) / float(self._total_num_cells)
      drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      # Decrease keep prob over time
      current_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
      current_ratio = tf.minimum(1.0, current_step / self._total_training_steps)
      drop_path_keep_prob = 1 - current_ratio * (1 - drop_path_keep_prob)
      # Drop path
      noise_shape = [net.shape[0], 1, 1, 1]
      random_tensor = drop_path_keep_prob
      random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
      binary_tensor = tf.cast(tf.floor(random_tensor), net.dtype)
      keep_prob_inv = tf.cast(1.0 / drop_path_keep_prob, net.dtype)
      net = net * keep_prob_inv * binary_tensor
    return net

  def _factorized_reduction(self, net, output_filters, stride):
    assert output_filters % 2 == 0
    if stride == 1:
      net = slim.conv2d(net, output_filters, 1, scope='path_conv')
      net = slim.batch_norm(net, scope='path_bn')
      return net
    stride_spec = [1, stride, stride, 1]

    # Skip path 1
    path1 = tf.nn.avg_pool(net, [1, 1, 1, 1], stride_spec, 'VALID')
    path1 = slim.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')

    # Skip path 2
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(net, pad_arr)[:, 1:, 1:, :]
    path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], stride_spec, 'VALID')
    path2 = slim.conv2d(path2, int(output_filters / 2), 1, scope='path2_conv')

    # Concat and apply BN
    final_path = tf.concat(values=[path1, path2], axis=3)
    final_path = slim.batch_norm(final_path, scope='final_path_bn')
    return final_path


class PNASCell(NASBaseCell):
  """PNASNet Cell."""

  def __init__(self, num_conv_filters, drop_path_keep_prob, total_num_cells,
               total_training_steps):
    # Configuration for the PNASNet-5 model.
    operations = [
        'separable_5x5_2', 'max_pool_3x3', 'separable_7x7_2', 'max_pool_3x3',
        'separable_5x5_2', 'separable_3x3_2', 'separable_3x3_2', 'max_pool_3x3',
        'separable_3x3_2', 'none'
    ]
    used_hiddenstates = [1, 1, 0, 0, 0, 0, 0]
    hiddenstate_indices = [1, 1, 0, 0, 0, 0, 4, 0, 1, 0]

    super(PNASCell, self).__init__(
        num_conv_filters, operations, used_hiddenstates, hiddenstate_indices,
        drop_path_keep_prob, total_num_cells, total_training_steps)
