from cores.utils.data import load_dataset, image_classes, load_parent_labels
import os
import tensorflow as tf


class DataSet(object):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1):
        self._num_gpus = num_gpus
        self._dataset = dataset
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._image_decoder = image_decoder
        self._num_epochs = num_epochs
        self._capacity = capacity
        self._num_threads = num_threads
        self._min_after_dequeue = min_after_dequeue
        self._shuffle = shuffle
        self._labels = image_classes(os.path.join(self._data_dir, 'train_val'))
        self._queue = self._batch_queue()

    def _input_tensors(self):
        image_dir, image_list, image_annotations = load_dataset(self._dataset, self._data_dir)
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

    def _enque_op(self, input_queue):
        enque_list = []
        for _ in range(self._num_threads):
            image_file = tf.read_file(input_queue[0])
            image = self._image_decoder.decode(image_file)
            label = tf.one_hot(input_queue[1], max(self._labels))
            enque_list.append([image, label])

        return enque_list

    def _batch_queue(self):
        tensors = self._input_tensors()
        input_queue = tf.train.slice_input_producer(tensors, num_epochs=self._num_epochs, shuffle=self._shuffle,
                                                    capacity=self._capacity)
        enque_op = self._enque_op(input_queue)
        inputs = tf.train.batch_join(
            enque_op, batch_size=self._batch_size,
            capacity=self._min_after_dequeue + (self._num_threads + 2) * self._batch_size,
        )
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            inputs, capacity=2 * self._num_gpus)
        return batch_queue

    def batch_inputs(self):
        with tf.device('/device:CPU:0'):
            inputs = self._queue.dequeue()
        return inputs


class DataSetWithParent(DataSet):

    def __init__(self, dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
                 num_threads=8, min_after_dequeue=1000, shuffle=True, num_gpus=1):
        self._parent_labels, self._child_to_parent = load_parent_labels(data_dir)
        super(DataSetWithParent, self).__init__(dataset, data_dir, batch_size, image_decoder, num_epochs, capacity,
                                                num_threads, min_after_dequeue, shuffle, num_gpus)

    def _input_tensors(self):
        image_dir, image_list, image_annotations = load_dataset(self._dataset, self._data_dir)
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
                tf_parent_list.append(self._child_to_parent[label_id])
        tf_images = tf.convert_to_tensor(tf_image_list, dtype=tf.string)
        tf_labels = tf.convert_to_tensor(tf_label_list, dtype=tf.int32)
        tf_parents = tf.convert_to_tensor(tf_parent_list, dtype=tf.int32)
        return tf_images, tf_labels, tf_parents

    def _enque_op(self, input_queue):
        enque_list = []
        for _ in range(self._num_threads):
            image_file = tf.read_file(input_queue[0])
            image = self._image_decoder.decode(image_file)
            label = tf.one_hot(input_queue[1], max(self._labels))
            parent = tf.one_hot(input_queue[2], len(self._parent_labels))
            enque_list.append([image, label, parent])
        return enque_list