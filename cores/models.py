from cores import InceptResV2, tf
from cores.utils.data import image_classes, load_dataset
from cores.utils.ops import softmax_accuracy_op
from tensorflow.contrib import slim
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


class FurnitureResV2Trainer(FurnitureResTrainer):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1,
                 keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        self.parent_labels, self.child_to_parent = self.load_parent_labels(data_dir)
        super(FurnitureResV2Trainer, self).__init__(dataset, data_dir, batch_size, image_decoder, num_epochs, capacity,
                                                    num_threads, min_after_dequeue, shuffle, num_gpus,
                                                    keep_prob, base_trainable, is_training,
                                                    weight_decay, batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/Parent_Logits")

    def load_parent_labels(self, data_dir):
        parent_labels = {}
        child_to_parent = {}
        with open(os.path.join(data_dir, 'parent_labels.json'), 'r') as f:
            parent_labels_dict = json.load(f)
        for i, key in enumerate(parent_labels_dict.keys()):
            parent_labels[key] = i
            for child in parent_labels_dict[key]:
                child_to_parent[child] = i
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
        logits = self._logits(net, max(self.labels), "Logits")
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