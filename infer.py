import os
import csv
import argparse
import tensorflow as tf
from datetime import datetime
from cores.models import FurnitureRes
from cores.utils.data import load_dataset, image_classes
from cores.utils.image import Decoder


FLAGS = None
tf.logging.set_verbosity(20)

def load_net(dataset_dir, model_path):
    labels = image_classes(os.path.join(dataset_dir, 'train_val'))
    sess = tf.Session()
    net = FurnitureRes()
    input_placehold = tf.placeholder(tf.string, [None])
    img_decoder = Decoder(299, 'default', 'minus_one_to_pos_one')
    image_inputs = tf.map_fn(img_decoder.decode, input_placehold, dtype=tf.float32)
    logits = net.output_logits(image_inputs, max(labels))
    predictions = tf.argmax(tf.nn.softmax(logits), 1)
    loader = tf.train.Saver()
    loader.restore(sess, model_path)
    return lambda a: sess.run(predictions, feed_dict={input_placehold: a})

def create_submission_csv(results, dataset_dir):
    curr = datetime.now()
    file_name = curr.strftime("%Y-%m-%dT%I_%M_%S") + '.csv'
    with open(os.path.join(dataset_dir, file_name), 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(('id', 'predicted'))
        for res in results:
            csv_writer.writerow(res)

def main():
    image_dir, image_list, image_annotations = load_dataset('test', FLAGS.dataset_dir)
    infer_fn = load_net(FLAGS.dataset_dir, FLAGS.model_path)
    total_images = len(image_list)
    res = []
    for i in range(0, total_images, 20):
        batch_images = image_list[i:i+20]
        image_inputs = []
        image_ids = []
        for j, image in enumerate(batch_images):
            image_path = os.path.join(image_dir, str(image['image_id']) + ".jpg")
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    byte_img = f.read()
                image_inputs.append(byte_img)
                image_ids.append(image['image_id'])
            else:
                res.append([image['image_id'], 0])
        tf.logging.info("Inference Images Between %d and %d" % (i, (i+20)))
        if image_inputs:
            predictions = infer_fn(image_inputs)
            # tf.logging.info(predictions)
            res.extend([(image_ids[k], pred) for k, pred in enumerate(predictions)])
    res = sorted(res, key=lambda a: a[0])
    # tf.logging.info(res)
    create_submission_csv(res, FLAGS.dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_dir',
        type=str,
    )
    parser.add_argument(
        'model_path',
        type=str,
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()