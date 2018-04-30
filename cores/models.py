from cores import InceptResV2, tf
from cores.utils.data import image_classes, load_dataset
from cores.utils.ops import softmax_accuracy_op
from tensorflow.contrib import slim
from cores.nets.inception_resnet_v2 import block8, block17, block35
import json
import os


class FurnitureResBuilder(InceptResV2):

    def __init__(self, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(FurnitureResBuilder, self).__init__(keep_prob, base_trainable, is_training, weight_decay,
                                                  batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/output_logits")
        self._logit_scope = "output_logits"


class FurnitureResTrainer(FurnitureResBuilder):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1,
                 keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):

        super(FurnitureResTrainer, self).__init__(keep_prob, base_trainable, is_training, weight_decay,
                                                  batch_norm_decay, batch_norm_epsilon)
        self._num_gpus = num_gpus
        self.labels = image_classes(os.path.join(data_dir, 'train_val'))
        inputs = self.batch_inputs(dataset, data_dir, batch_size, image_decoder,
                                   num_epochs, capacity, num_threads,
                                   min_after_dequeue, shuffle)
        self._batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            inputs, capacity=2 * num_gpus)

    def batch_inputs(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                     num_threads=8, min_after_dequeue=1000, shuffle=True):
        image_dir, image_list, image_annotations = load_dataset(dataset, data_dir)
        tensors = self.input_tensors(image_dir, image_list, image_annotations)
        input_queue = tf.train.slice_input_producer(tensors, num_epochs=num_epochs, shuffle=shuffle,
                                                    capacity=capacity)
        enque_op = self.enque_op(image_decoder, num_threads, input_queue)
        inputs = tf.train.batch_join(
            enque_op, batch_size=batch_size,
            capacity=min_after_dequeue + (num_threads + 2) * batch_size,
        )
        return inputs

    def input_tensors(self, image_dir, image_list, image_annotations):
        tf_image_list = []
        tf_label_list = []
        for i, image in enumerate(image_list):
            image_id = image['image_id']
            label_id = image_annotations[i]['label_id']
            image_path = os.path.join(image_dir, str(label_id), str(image_id) + '.jpg')
            if os.path.exists(image_path):
                tf_image_list.append(image_path)
                tf_label_list.append(label_id)
        tf_images = tf.convert_to_tensor(tf_image_list, dtype=tf.string)
        tf_labels = tf.convert_to_tensor(tf_label_list, dtype=tf.int32)
        return tf_images, tf_labels

    def enque_op(self, image_decoder, num_threads, input_queue):

        enque_list = []
        for _ in range(num_threads):
            image_file = tf.read_file(input_queue[0])
            image = image_decoder.decode(image_file)
            label = tf.one_hot(input_queue[1], max(self.labels))
            enque_list.append([image, label])

        return enque_list

    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        logits = self._logits(net, max(self.labels), "Logits")
        return logits

    def net_loss(self, reuse=tf.AUTO_REUSE, reg_scope=None):
        with tf.device('/device:CPU:0'):
            batch_images, batch_labels = self._batch_queue.dequeue()
        logits = self.output_logits(batch_images, reuse=reuse)
        cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='cross_entropy')
        res_loss = tf.reduce_mean(cross_entropy, name='loss')

        total_loss = tf.add_n([res_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope))
        res_accuracy = softmax_accuracy_op(logits, batch_labels)
        with tf.device('/device:CPU:0'):
            tf.summary.scalar("loss", res_loss)
            tf.summary.scalar('accuracy', res_accuracy)
        return total_loss

    def net_eval(self):
        with tf.device('/device:CPU:0'):
            batch_images, batch_labels = self._batch_queue.dequeue()
        logits = self.output_logits(batch_images, max(self.labels))
        predictions = tf.argmax(tf.nn.softmax(logits), 1)

        labels = tf.argmax(batch_labels, 1)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': tf.metrics.accuracy(predictions=predictions, labels=labels),
        })
        return names_to_values, names_to_updates


class FurnitureResParentTrainer(FurnitureResTrainer):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1,
                 keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        self.parent_labels, self.child_to_parent = self.load_parent_labels(data_dir)
        super(FurnitureResParentTrainer, self).__init__(dataset, data_dir, batch_size, image_decoder, num_epochs, capacity,
                                                    num_threads, min_after_dequeue, shuffle, num_gpus,
                                                    keep_prob, base_trainable, is_training,
                                                    weight_decay, batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/Parent_Logits")

    def load_parent_labels(self, data_dir):
        parent_labels = {}
        child_to_parent = {}
        with open(os.path.join(data_dir, 'parent_labels.json'), 'r') as f:
            parent_labels_dict = json.load(f)
        for parent_label_name, parent_values in parent_labels_dict.items():
            parent_labels[parent_label_name] = parent_values['id']
            for child in parent_values['subs']:
                child_to_parent[child] = parent_values['id']
        return parent_labels, child_to_parent

    def input_tensors(self, image_dir, image_list, image_annotations):
        tf_image_list = []
        tf_label_list = []
        tf_parent_list = []
        for i, image in enumerate(image_list):
            image_id = image['image_id']
            label_id = image_annotations[i]['label_id']
            image_path = os.path.join(image_dir, str(label_id), str(image_id) + '.jpg')
            if os.path.exists(image_path):
                tf_image_list.append(image_path)
                tf_label_list.append(label_id)
                tf_parent_list.append(self.child_to_parent[label_id])
        tf_images = tf.convert_to_tensor(tf_image_list, dtype=tf.string)
        tf_labels = tf.convert_to_tensor(tf_label_list, dtype=tf.int32)
        tf_parents = tf.convert_to_tensor(tf_parent_list, dtype=tf.int32)
        return tf_images, tf_labels, tf_parents

    def enque_op(self, image_decoder, num_threads, input_queue):
        enque_list = []
        for _ in range(num_threads):
            image_file = tf.read_file(input_queue[0])
            image = image_decoder.decode(image_file)
            label = tf.one_hot(input_queue[1], max(self.labels))
            parent = tf.one_hot(input_queue[2], len(self.parent_labels))
            enque_list.append([image, label, parent])
        return enque_list

    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        parent_logits = self._logits(net, len(self.parent_labels), "Parent_Logits")
        return parent_logits

    def net_loss(self, reuse=tf.AUTO_REUSE, reg_scope=None):
        with tf.device('/device:CPU:0'):
            batch_images, batch_labels, batch_parents = self._batch_queue.dequeue()
        parent_logits = self.output_logits(batch_images, reuse=reuse)

        parent_cross_entropy = tf.losses.softmax_cross_entropy(batch_parents, parent_logits, scope='parent_cross_entropy')
        parent_loss = tf.reduce_mean(parent_cross_entropy, name='parent_loss')

        total_loss = tf.add_n([parent_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope))
        parent_accuracy = softmax_accuracy_op(parent_logits, batch_parents)
        with tf.device('/device:CPU:0'):
            tf.summary.scalar('parent_loss', parent_loss)
            tf.summary.scalar('parent_accuracy', parent_accuracy)
        return total_loss

    def net_eval(self):
        with tf.device('/device:CPU:0'):
            batch_images, batch_labels, batch_parents = self._batch_queue.dequeue()
        parent_logits = self.output_logits(batch_images)

        parent_predictions = tf.argmax(tf.nn.softmax(parent_logits), 1)
        parent_labels = tf.argmax(batch_parents, 1)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Parent_Accuracy': tf.metrics.accuracy(predictions=parent_predictions, labels=parent_labels),
        })
        return names_to_values, names_to_updates


class FurnitureResMixedTrainer(FurnitureResParentTrainer):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1,
                 keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(FurnitureResMixedTrainer, self).__init__(dataset, data_dir, batch_size, image_decoder, num_epochs, capacity,
                                                    num_threads, min_after_dequeue, shuffle, num_gpus,
                                                    keep_prob, base_trainable, is_training,
                                                    weight_decay, batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/Child_Logits")

    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        parent_logits = self._logits(net, len(self.parent_labels), "Parent_Logits")
        logits = self._logits(net, max(self.labels), "Child_Logits")
        return parent_logits, logits

    def net_loss(self, reuse=tf.AUTO_REUSE, reg_scope=None):
        with tf.device('/device:CPU:0'):
            batch_images, batch_labels, batch_parents = self._batch_queue.dequeue()
        parent_logits, logits = self.output_logits(batch_images, reuse=reuse)

        child_cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='child_cross_entropy')
        child_loss = tf.reduce_mean(child_cross_entropy, name='child_loss')

        parent_cross_entropy = tf.losses.softmax_cross_entropy(batch_parents, parent_logits, scope='parent_cross_entropy')
        parent_loss = tf.reduce_mean(parent_cross_entropy, name='parent_loss')

        total_loss = tf.add_n([parent_loss, child_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope))
        child_accuracy = softmax_accuracy_op(logits, batch_labels)
        parent_accuracy = softmax_accuracy_op(parent_logits, batch_parents)
        with tf.device('/device:CPU:0'):
            tf.summary.scalar('child_loss', child_loss)
            tf.summary.scalar('parent_loss', parent_loss)
            tf.summary.scalar('child_accuracy', child_accuracy)
            tf.summary.scalar('parent_accuracy', parent_accuracy)
        return total_loss

    def net_eval(self):
        with tf.device('/device:CPU:0'):
            batch_images, batch_labels, batch_parents = self._batch_queue.dequeue()
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



class FurnitureAttentionTrainer(FurnitureResMixedTrainer):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1,
                 keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        self.parent_labels, self.child_to_parent = self.load_parent_labels(data_dir)
        super(FurnitureResMixedTrainer, self).__init__(dataset, data_dir, batch_size, image_decoder, num_epochs, capacity,
                                                       num_threads, min_after_dequeue, shuffle, num_gpus,
                                                       keep_prob, base_trainable, is_training,
                                                       weight_decay, batch_norm_decay, batch_norm_epsilon)
        self._attention_trainable = is_training
        self._hidden_size = 320
        self._num_layers = 1
        self.prelogits_names = ["InceptionAttention", "InceptionChild"]

    def _build_rnn_net(self, batch_size, initial_embedding):
        def _cell():
            cell = tf.contrib.rnn.LSTMCell(self._hidden_size)
            if self._is_training and self._keep_prob < 1.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self._keep_prob, output_keep_prob=self._keep_prob)
            return cell

        with tf.variable_scope('lstm', initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08),
                               regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                               reuse=tf.AUTO_REUSE) as lstm_scope:
            lstm_cell = tf.contrib.rnn.MultiRNNCell([_cell() for _ in range(self._num_layers)], state_is_tuple=True)
            zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(initial_embedding, zero_state)
        return lstm_cell, initial_state

    def _build_child_net(self, net):
        padding = 'VALID'
        with slim.arg_scope(self._arg_scope):

            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm],
                                trainable=self._attention_trainable):
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self._is_training):
                    net = slim.repeat(net, 10, block35, scale=0.17)

                    # 17 x 17 x 1088 if output_stride == 8,
                    # 33 x 33 x 1088 if output_stride == 16
                    use_atrous = True

                    with tf.variable_scope('Mixed_6a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                                     padding=padding,
                                                     scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                        scope='Conv2d_0b_3x3')
                            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                        stride=1 if use_atrous else 2,
                                                        padding=padding,
                                                        scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                                         padding=padding,
                                                         scope='MaxPool_1a_3x3')
                        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

                    # TODO(alemi): register intermediate endpoints
                    with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
                        net = slim.repeat(net, 20, block17, scale=0.10)

                    # 8 x 8 x 2080
                    with tf.variable_scope('Mixed_7a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                       padding=padding,
                                                       scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                        padding=padding,
                                                        scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                        scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                        padding=padding,
                                                        scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2,
                                                         padding=padding,
                                                         scope='MaxPool_1a_3x3')
                        net = tf.concat(
                            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

                    # TODO(alemi): register intermediate endpoints
                    net = slim.repeat(net, 9, block8, scale=0.20)
                    net = block8(net, activation_fn=None)

                    # 8 x 8 x 1536
                    net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')

                    with tf.variable_scope('Logits'):
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                              scope='AvgPool_1a_8x8')
                        net = slim.flatten(net)

                        net = slim.dropout(net, self._keep_prob, is_training=self._is_training,
                                           scope='Dropout')

                        logits = slim.fully_connected(net, self.labels, activation_fn=None,
                                                      scope='Logits')
        return logits


    def output_logits(self, inputs, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        parent_logits = self._logits(net, len(self.parent_labels), "Parent_Logits")

        attention_field_map = end_points['Mixed_5b']
        field_size = 35
        attention_fields = []
        for i in range(field_size):
            for j in range(field_size):
                attention_fields.append(attention_field_map[:, i, j, :])
        attention_fields = tf.stack(attention_fields, axis=0)
        with tf.variable_scope('InceptionAttention', 'InceptionAttention', [attention_fields]) as attention_scope:
            with tf.variable_scope("attention_transform") as transform_scope :
                attention_embedding = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=320,
                    activation_fn=None,
                    scope=transform_scope
                )
            lstm_cell, init_state = self._build_rnn_net(inputs.shape[0], attention_embedding)
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=attention_fields,
                                                initial_state=init_state,
                                                time_major=True,
                                                dtype=tf.float32,
                                                scope=attention_scope)
            with tf.variable_scope("attention_logits") as logits_scope:
                rnn_gates = tf.contrib.layers.fully_connected(
                    inputs=lstm_outputs[-1],
                    num_outputs=1225,
                    activation_fn=tf.nn.sigmoid,
                    scope=logits_scope)
            rnn_gates = tf.reshape(rnn_gates, [inputs.shape[0], field_size, field_size, 1])
            gated_field = tf.multiply(attention_field_map, rnn_gates)

        with tf.variable_scope('InceptionChild', 'InceptionChild', [gated_field]):
            child_logits = self._build_child_net(gated_field)

        return parent_logits, child_logits