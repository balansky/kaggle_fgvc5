import os
import json


def load_dataset(dataset, dataset_dir):

    with open(os.path.join(dataset_dir, dataset + '.json'), 'r') as f:
        metas = json.load(f)

    if dataset == 'train' or dataset == 'validation' or dataset == 'val':

        image_list, image_annotations = metas['images'], metas['annotations']
        image_dir = os.path.join(dataset_dir, "train_val")

    else:
        image_list = metas['images']
        image_annotations = None
        image_dir = os.path.join(dataset_dir, "test")
    return image_dir, image_list, image_annotations


def load_parent_labels(data_dir):
    parent_labels = {}
    child_to_parent = {}
    with open(os.path.join(data_dir, 'parent_labels.json'), 'r') as f:
        parent_labels_dict = json.load(f)
    for parent_label_name, parent_values in parent_labels_dict.items():
        parent_labels[parent_label_name] = parent_values['id']
        for child in parent_values['subs']:
            child_to_parent[child] = parent_values['id']
    return parent_labels, child_to_parent


def load_child_labels(data_dir):
    child_to_orig = {}
    with open(os.path.join(data_dir, 'child_labels.json'), 'r') as f:
       org_to_child = json.load(f)
    for c_key, c_value in org_to_child.items():
        if c_value not in child_to_orig:
            child_to_orig[str(c_value)] = c_key
    return org_to_child, child_to_orig


def image_classes(image_dir):
    classes = [int(path) for path in os.listdir(image_dir)
               if os.path.isdir(os.path.join(image_dir, path))]

    classes = sorted(classes)
    return classes


# def input_tensors(image_dir, image_list, image_annotations):
#     tf_image_list = []
#     tf_label_list = []
#     for i, image in enumerate(image_list):
#         image_id = image['image_id']
#         label_id = image_annotations[i]['label_id']
#         image_path = os.path.join(image_dir, str(label_id), str(image_id) + '.jpg')
#         if os.path.exists(image_path):
#             tf_image_list.append(image_path)
#             tf_label_list.append(label_id)
#     tf_images = tf.convert_to_tensor(tf_image_list, dtype=tf.string)
#     tf_labels = tf.convert_to_tensor(tf_label_list, dtype=tf.int32)
#     return tf_images, tf_labels
#
#
#
#
# def batch_inputs(dataset, data_dir, batch_size, image_decoder, num_epochs=None, capacity=60,
#                  num_threads=8, min_after_dequeue=1000, shuffle=True):
#     image_dir, image_list, image_annotations = load_dataset(dataset, data_dir)
#     labels = image_classes(image_dir)
#     image_list, label_list = input_tensors(image_dir, image_list, image_annotations)
#     input_queue = tf.train.slice_input_producer([image_list, label_list], num_epochs=num_epochs, shuffle=shuffle,
#                                                 capacity=capacity)
#     enque_op = []
#     for _ in range(num_threads):
#         image_file = tf.read_file(input_queue[0])
#         image = image_decoder.decode(image_file)
#         label = tf.one_hot(input_queue[1], max(labels))
#         enque_op.append([image, label])
#
#     image_batch, label_batch = tf.train.batch_join(
#         enque_op, batch_size=batch_size,
#         capacity=min_after_dequeue + (num_threads + 2) * batch_size,
#         )
#     return image_batch, label_batch
#

