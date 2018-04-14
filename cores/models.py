from tensorflow.contrib import slim
import tensorflow as tf
from cores import inception_v4, inception_resnet_v2


class SlimNet(object):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False, reuse=tf.AUTO_REUSE):
        self._scope_name = None
        self._prelogits_names = []
        self._inputs = inputs
        self._keep_prob = keep_prob
        self._cnn_trainable = base_trainable
        self._is_training = is_training
        self._reuse = reuse

    @property
    def prelogits_names(self):
        return self._prelogits_names

    @prelogits_names.setter
    def prelogits_names(self, value):
        self._prelogits_names = value

    def variables_without_prelogits(self):
        scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope_name)
        if self._prelogits_names:
            variables_to_restore = []
            for var in scope_vars:
                excluded = False
                for exclusion in self._prelogits_names:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)
        else:
            variables_to_restore = scope_vars
        return variables_to_restore

    def restore_fn(self, ckpt_dir, pretrained_ckpt=None):
        try:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            if not ckpt_path and pretrained_ckpt:
                ckpt_path = pretrained_ckpt
                restore_variables = self.variables_without_prelogits()
            else:
                restore_variables = tf.trainable_variables()
            return slim.assign_from_checkpoint_fn(ckpt_path, restore_variables, ignore_missing_vars=False)
        except Exception as err:
            tf.logging.warning(err)
            return None


class InceptV4(SlimNet):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False,
                 use_batch_norm=True, weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001,
                 reuse=tf.AUTO_REUSE):
        super(InceptV4, self).__init__(inputs, keep_prob, base_trainable, is_training, reuse)
        self._prelogits_names = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        self._scope_name = 'InceptionV4'
        self._use_batch_norm = use_batch_norm
        self._batch_norm_decay = batch_norm_decay
        self._weight_decay = weight_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self.net, self.end_points = self._build_net()

    def _build_net(self):

        with slim.arg_scope(inception_v4.inception_v4_arg_scope(use_batch_norm=self._use_batch_norm,
                                                                weight_decay=self._weight_decay,
                                                                batch_norm_decay=self._batch_norm_decay,
                                                                batch_norm_epsilon=self._batch_norm_epsilon)):
            with tf.variable_scope(self._scope_name, 'InceptionV4', reuse=self._reuse) as scope:
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm], trainable=self._cnn_trainable):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self._cnn_trainable):
                        net, end_points = inception_v4.inception_v4_base(self._inputs, scope=scope)
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a')
        return net, end_points

    def pre_logits(self):
        return slim.flatten(self.net, scope='logits')

    def output_logits(self, num_labels, net=None, scope='Logits'):

        net = net if net is not None else self.net

        with slim.arg_scope(inception_v4.inception_v4_arg_scope(use_batch_norm=self._use_batch_norm,
                                                                weight_decay=self._weight_decay,
                                                                batch_norm_decay=self._batch_norm_decay,
                                                                batch_norm_epsilon=self._batch_norm_epsilon)):
            with tf.variable_scope(self._scope_name, 'InceptionV4'):
                with tf.variable_scope(scope):
                    logits = slim.dropout(net, keep_prob=self._keep_prob, is_training=self._is_training,
                                       scope='Dropout_1b')
                    logits = slim.flatten(logits, scope='PreLogitsFlatten')
                    logits = slim.fully_connected(logits, num_labels, activation_fn=None,
                                                  normalizer_fn=None, scope='Logits')
        return logits


class InceptResV2(SlimNet):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, reuse=tf.AUTO_REUSE):
        super(InceptResV2, self).__init__(inputs, keep_prob, base_trainable, is_training, reuse)
        self._prelogits_names = ["InceptionResnetV2/AuxLogits", "InceptionResnetV2/Logits"]
        self._scope_name = 'InceptionResnetV2'
        self._batch_norm_decay = batch_norm_decay
        self._weight_decay = weight_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope(
                                        weight_decay=self._weight_decay,
                                        batch_norm_decay=self._batch_norm_decay,
                                        batch_norm_epsilon=self._batch_norm_epsilon
        )
        self.net, self.end_points = self._build_net()

    def _build_net(self):

        with slim.arg_scope(self._arg_scope):
            with tf.variable_scope(self._scope_name, 'InceptionResnetV2', reuse=self._reuse) as scope:
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm], trainable=self._cnn_trainable):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self._cnn_trainable):
                        net, end_points = inception_resnet_v2.inception_resnet_v2_base(self._inputs, scope=scope)
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a')
        return net, end_points

    def pre_logits(self):
        return slim.flatten(self.net, scope='logits')

    def output_logits(self, num_labels, net=None, scope='Logits'):

        net = net if net is not None else self.net

        with slim.arg_scope(self._arg_scope):

            with tf.variable_scope(self._scope_name, 'InceptionResnetV2'):
                with tf.variable_scope(scope):
                    logits = slim.dropout(net, keep_prob=self._keep_prob, is_training=self._is_training,
                                       scope='Dropout_1b')
                    logits = slim.flatten(logits, scope='PreLogitsFlatten')
                    logits = slim.fully_connected(logits, num_labels, activation_fn=None,
                                                  normalizer_fn=None, scope='Logits')
        return logits