from cores import InceptResV2, tf
from cores.utils.data import load_child_labels, load_parent_labels
from cores.utils.ops import softmax_accuracy_op
from tensorflow.contrib import slim
from cores.nets.inception_resnet_v2 import block8, block17, block35
import json
import os


class FurnitureInceptRes(InceptResV2):

    def __init__(self, data_dir, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(FurnitureInceptRes, self).__init__(keep_prob, base_trainable, is_training, weight_decay,
                                                  batch_norm_decay, batch_norm_epsilon)
        self._orig_to_child, self._child_to_orig = load_child_labels(data_dir)
        self.labels = [k for k in self._child_to_orig.keys()]
        self.prelogits_names.append("InceptionResnetV2/output_logits")
        self._logit_scope = "output_logits"


    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        logits = self._logits(net, len(self.labels), "Logits")
        return logits

    def net_loss(self, batch_inputs, reuse=tf.AUTO_REUSE, reg_scope=None):
        batch_images, batch_labels = batch_inputs
        logits = self.output_logits(batch_images, reuse=reuse)
        cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='cross_entropy')
        res_loss = tf.reduce_mean(cross_entropy, name='loss')
        if not self._cnn_trainable:
            regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     scope=reg_scope + "/InceptionResnetV2/output_logits")
        else:
            regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope)
        total_loss = tf.add_n([res_loss] + regs)
        res_accuracy = softmax_accuracy_op(logits, batch_labels)
        with tf.device('/device:CPU:0'):
            tf.summary.scalar("loss", total_loss)
            tf.summary.scalar("softmax_loss", res_loss)
            tf.summary.scalar('accuracy', res_accuracy)
        return total_loss

    def net_eval(self, batch_inputs):
        batch_images, batch_labels = batch_inputs
        logits = self.output_logits(batch_images)
        predictions = tf.argmax(tf.nn.softmax(logits), 1)

        labels = tf.argmax(batch_labels, 1)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': tf.metrics.accuracy(predictions=predictions, labels=labels),
        })
        return names_to_values, names_to_updates


class FurnitureInceptResParent(FurnitureInceptRes):

    def __init__(self, data_dir, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        self.parent_labels, self.child_to_parent = load_parent_labels(data_dir)
        super(FurnitureInceptResParent, self).__init__(data_dir,
                                                 keep_prob, base_trainable, is_training,
                                                 weight_decay, batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/Parent_Logits")


    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        parent_logits = self._logits(net, len(self.parent_labels), "Parent_Logits")
        return parent_logits

    def net_loss(self, batch_inputs, reuse=tf.AUTO_REUSE, reg_scope=None):
        batch_images, batch_labels, batch_parents = batch_inputs
        parent_logits = self.output_logits(batch_images, reuse=reuse)

        parent_cross_entropy = tf.losses.softmax_cross_entropy(batch_parents, parent_logits, scope='parent_cross_entropy')
        parent_loss = tf.reduce_mean(parent_cross_entropy, name='parent_loss')

        if not self._cnn_trainable:
            regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     scope=reg_scope + "/InceptionResnetV2/Parent_Logits")
        else:
            regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope)

        total_loss = tf.add_n([parent_loss] + regs)
        parent_accuracy = softmax_accuracy_op(parent_logits, batch_parents)
        with tf.device('/device:CPU:0'):
            tf.summary.scalar('parent_loss', parent_loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('parent_accuracy', parent_accuracy)
        return total_loss

    def net_eval(self, batch_inputs):
        batch_images, batch_labels, batch_parents = batch_inputs
        parent_logits = self.output_logits(batch_images)

        parent_predictions = tf.argmax(tf.nn.softmax(parent_logits), 1)
        parent_labels = tf.argmax(batch_parents, 1)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Parent_Accuracy': tf.metrics.accuracy(predictions=parent_predictions, labels=parent_labels),
        })
        return names_to_values, names_to_updates


class FurnitureInceptResMixed(FurnitureInceptResParent):

    def __init__(self, data_dir, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(FurnitureInceptResMixed, self).__init__(data_dir,
                                                       keep_prob, base_trainable, is_training,
                                                       weight_decay, batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/Child_Logits")

    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        parent_logits = self._logits(net, len(self.parent_labels), "Parent_Logits")
        logits = self._logits(net, len(self.labels), "Child_Logits")
        return parent_logits, logits

    def net_loss(self, batch_inputs, reuse=tf.AUTO_REUSE, reg_scope=None):
        batch_images, batch_labels, batch_parents = batch_inputs
        parent_logits, logits = self.output_logits(batch_images, reuse=reuse)

        child_cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='child_cross_entropy')
        child_loss = tf.reduce_mean(child_cross_entropy, name='child_loss')

        parent_cross_entropy = tf.losses.softmax_cross_entropy(batch_parents, parent_logits, scope='parent_cross_entropy')
        parent_loss = tf.reduce_mean(parent_cross_entropy, name='parent_loss')

        if not self._cnn_trainable:
            child_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                           scope=reg_scope + "/InceptionResnetV2/Child_Logits")
            parent_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                            scope=reg_scope + "/InceptionResnetV2/Parent_Logits")
            regs = child_regs + parent_regs
        else:
            regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope)
        total_loss = tf.add_n([parent_loss, child_loss] + regs)
        child_accuracy = softmax_accuracy_op(logits, batch_labels)
        parent_accuracy = softmax_accuracy_op(parent_logits, batch_parents)
        with tf.device('/device:CPU:0'):
            tf.summary.scalar('child_loss', child_loss)
            tf.summary.scalar('parent_loss', parent_loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('child_accuracy', child_accuracy)
            tf.summary.scalar('parent_accuracy', parent_accuracy)
        return total_loss

    def net_eval(self, batch_inputs):
        batch_images, batch_labels, batch_parents = batch_inputs
        parent_logits, logits = self.output_logits(batch_images)

        parent_predictions = tf.argmax(tf.nn.softmax(parent_logits), 1)
        parent_labels = tf.argmax(batch_parents, 1)

        child_predictions = tf.argmax(tf.nn.softmax(logits), 1)
        child_labels = tf.argmax(batch_labels, 1)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Parent_Accuracy': tf.metrics.accuracy(predictions=parent_predictions, labels=parent_labels),
            'Child_Accuracy': tf.metrics.accuracy(predictions=child_predictions, labels=child_labels),
        })
        return names_to_values, names_to_updates



# class FurnitureAttentionTrainer(FurnitureResMixedTrainer):
#
#     def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
#                  num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1,
#                  keep_prob=1.0, base_trainable=False, is_training=False,
#                  weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
#         self.parent_labels, self.child_to_parent = self.load_parent_labels(data_dir)
#         super(FurnitureResMixedTrainer, self).__init__(dataset, data_dir, batch_size, image_decoder, num_epochs, capacity,
#                                                        num_threads, min_after_dequeue, shuffle, num_gpus,
#                                                        keep_prob, base_trainable, is_training,
#                                                        weight_decay, batch_norm_decay, batch_norm_epsilon)
#         self._attention_trainable = is_training
#         self._hidden_size = 320
#         self._num_layers = 1
#         self.prelogits_names = ["InceptionAttention", "InceptionChild"]
#
#     def _build_rnn_net(self, batch_size, initial_embedding):
#         def _cell():
#             cell = tf.contrib.rnn.LSTMCell(self._hidden_size)
#             if self._is_training and self._keep_prob < 1.0:
#                 cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self._keep_prob, output_keep_prob=self._keep_prob)
#             return cell
#
#         with tf.variable_scope('lstm', initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08),
#                                regularizer=tf.contrib.layers.l2_regularizer(0.0004),
#                                reuse=tf.AUTO_REUSE) as lstm_scope:
#             lstm_cell = tf.contrib.rnn.MultiRNNCell([_cell() for _ in range(self._num_layers)], state_is_tuple=True)
#             zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
#             _, initial_state = lstm_cell(initial_embedding, zero_state)
#         return lstm_cell, initial_state
#
#     def _build_child_net(self, net):
#         padding = 'VALID'
#         with slim.arg_scope(self._arg_scope):
#
#             with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm],
#                                 trainable=self._attention_trainable):
#                 with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self._is_training):
#                     net = slim.repeat(net, 10, block35, scale=0.17)
#
#                     # 17 x 17 x 1088 if output_stride == 8,
#                     # 33 x 33 x 1088 if output_stride == 16
#                     use_atrous = True
#
#                     with tf.variable_scope('Mixed_6a'):
#                         with tf.variable_scope('Branch_0'):
#                             tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
#                                                      padding=padding,
#                                                      scope='Conv2d_1a_3x3')
#                         with tf.variable_scope('Branch_1'):
#                             tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#                             tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
#                                                         scope='Conv2d_0b_3x3')
#                             tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
#                                                         stride=1 if use_atrous else 2,
#                                                         padding=padding,
#                                                         scope='Conv2d_1a_3x3')
#                         with tf.variable_scope('Branch_2'):
#                             tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
#                                                          padding=padding,
#                                                          scope='MaxPool_1a_3x3')
#                         net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
#
#                     # TODO(alemi): register intermediate endpoints
#                     with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
#                         net = slim.repeat(net, 20, block17, scale=0.10)
#
#                     # 8 x 8 x 2080
#                     with tf.variable_scope('Mixed_7a'):
#                         with tf.variable_scope('Branch_0'):
#                             tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#                             tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
#                                                        padding=padding,
#                                                        scope='Conv2d_1a_3x3')
#                         with tf.variable_scope('Branch_1'):
#                             tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#                             tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
#                                                         padding=padding,
#                                                         scope='Conv2d_1a_3x3')
#                         with tf.variable_scope('Branch_2'):
#                             tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
#                             tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
#                                                         scope='Conv2d_0b_3x3')
#                             tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
#                                                         padding=padding,
#                                                         scope='Conv2d_1a_3x3')
#                         with tf.variable_scope('Branch_3'):
#                             tower_pool = slim.max_pool2d(net, 3, stride=2,
#                                                          padding=padding,
#                                                          scope='MaxPool_1a_3x3')
#                         net = tf.concat(
#                             [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
#
#                     # TODO(alemi): register intermediate endpoints
#                     net = slim.repeat(net, 9, block8, scale=0.20)
#                     net = block8(net, activation_fn=None)
#
#                     # 8 x 8 x 1536
#                     net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
#
#                     with tf.variable_scope('Logits'):
#                         net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
#                                               scope='AvgPool_1a_8x8')
#                         net = slim.flatten(net)
#
#                         net = slim.dropout(net, self._keep_prob, is_training=self._is_training,
#                                            scope='Dropout')
#
#                         logits = slim.fully_connected(net, len(self.labels), activation_fn=None,
#                                                       scope='Logits')
#         return logits
#
#
#     def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
#         net, end_points = self.build_net(inputs, reuse)
#         parent_logits = self._logits(net, len(self.parent_labels), "Parent_Logits", trainable=False)
#
#
#         attention_field_map = end_points['Mixed_5b']
#         gated_field = attention_field_map
#         # field_size = 35
#         # attention_fields = []
#         # for i in range(field_size):
#         #     for j in range(field_size):
#         #         attention_fields.append(attention_field_map[:, i, j, :])
#         # attention_fields = tf.stack(attention_fields, axis=0)
#         # with tf.variable_scope('InceptionAttention', 'InceptionAttention', [attention_fields]) as attention_scope:
#         #     with tf.variable_scope("attention_transform") as transform_scope :
#         #         attention_embedding = tf.contrib.layers.fully_connected(
#         #             inputs=net,
#         #             num_outputs=self._hidden_size,
#         #             activation_fn=None,
#         #             scope=transform_scope
#         #         )
#         #     lstm_cell, init_state = self._build_rnn_net(inputs.shape[0], attention_embedding)
#         #     lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
#         #                                         inputs=attention_fields,
#         #                                         initial_state=init_state,
#         #                                         time_major=True,
#         #                                         dtype=tf.float32,
#         #                                         scope=attention_scope)
#         #     lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
#         #     gated_field = tf.reshape(lstm_outputs, [inputs.shape[0], field_size, field_size, self._hidden_size])
#             # with tf.variable_scope("attention_logits") as logits_scope:
#             #     rnn_gates = tf.contrib.layers.fully_connected(
#             #         inputs=lstm_outputs[-1],
#             #         num_outputs=1225,
#             #         activation_fn=tf.nn.sigmoid,
#             #         scope=logits_scope)
#             # rnn_gates = tf.reshape(rnn_gates, [inputs.shape[0], field_size, field_size, 1])
#             # gated_field = tf.multiply(attention_field_map, rnn_gates)
#
#         with tf.variable_scope('InceptionChild', 'InceptionChild', [gated_field]):
#             child_logits = self._build_child_net(gated_field)
#
#         return parent_logits, child_logits
#
#     def net_loss(self, reuse=tf.AUTO_REUSE, reg_scope=None):
#         with tf.device('/device:CPU:0'):
#             batch_images, batch_labels, batch_parents = self._batch_queue.dequeue()
#         parent_logits, logits = self.output_logits(batch_images, reuse=reuse)
#
#         losses = []
#         child_cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='child_cross_entropy')
#         child_loss = tf.reduce_mean(child_cross_entropy, name='child_loss')
#
#         losses.append(child_loss)
#         if self._attention_trainable:
#             parent_cross_entropy = tf.losses.softmax_cross_entropy(batch_parents, parent_logits, scope='parent_cross_entropy')
#             parent_loss = tf.reduce_mean(parent_cross_entropy, name='parent_loss')
#             losses.append(parent_loss)
#         if not self._cnn_trainable:
#             reg_scope = reg_scope + '/InceptionChild'
#
#         total_loss = tf.add_n(losses + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope))
#         child_accuracy = softmax_accuracy_op(logits, batch_labels)
#         with tf.device('/device:CPU:0'):
#             tf.summary.scalar('child_loss', child_loss)
#             tf.summary.scalar('child_accuracy', child_accuracy)
#         return total_loss