import os
import argparse
import tensorflow as tf
from tensorflow.contrib import slim
from cores.models import InceptResV2, InceptV4
from utils import Decoder, train_op, image_classes, batch_inputs

FLAGS = None
tf.logging.set_verbosity(20)


def _train(model, sess_config):

    img_decoder = Decoder(FLAGS.input_size, FLAGS.resize_method, FLAGS.norm_method, random_flip=FLAGS.random_flip,
                          distort_color=FLAGS.distort_color)
    batch_images, batch_labels = batch_inputs('train', FLAGS.dataset_dir, FLAGS.batch_size, img_decoder,
                                              FLAGS.num_epochs, FLAGS.capacity, FLAGS.num_threads,
                                              FLAGS.min_after_dequeue)
    labels = image_classes(os.path.join(FLAGS.dataset_dir, 'train_val'))
    net = model(batch_images, keep_prob=FLAGS.keep_prob, base_trainable=FLAGS.base_trainable,
                is_training=True)
    net.prelogits_names.append(["InceptionResnetV2/output_logits"])
    logits = net.output_logits(max(labels), scope='output_logits')

    res_softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(res_softmax, 1)
    label_idx = tf.argmax(batch_labels, 1)
    correct_prediction = tf.equal(predictions, label_idx)
    res_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', res_accuracy)

    cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='cross_entropy')
    res_loss = tf.reduce_mean(cross_entropy, name='loss')

    tf.summary.scalar('loss', res_loss)

    total_loss = tf.add_n([res_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    train_step, global_step = train_op(total_loss, FLAGS.learning_rate, FLAGS.optimizer, FLAGS.decay_frequency,
                                       FLAGS.decay_rate)

    saver = tf.train.Saver(max_to_keep=5)
    summary_op = tf.summary.merge_all()
    summary_dir = os.path.join(FLAGS.ckpt_dir, 'summary', FLAGS.summary_name)
    summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
    slim.learning.train(train_step, FLAGS.ckpt_dir,
                        session_config=sess_config,
                        init_fn=net.restore_fn(FLAGS.ckpt_dir, FLAGS.pretrained_ckpt),
                        log_every_n_steps=10,
                        global_step=global_step, number_of_steps=FLAGS.num_steps, summary_op=summary_op,
                        summary_writer=summary_writer, saver=saver, save_interval_secs=60, save_summaries_secs=60)


def _eval(model, sess_config):

    img_decoder = Decoder(FLAGS.input_size, FLAGS.resize_method, FLAGS.norm_method)
    batch_images, batch_labels = batch_inputs('validation', FLAGS.dataset_dir, FLAGS.batch_size, img_decoder,
                                              FLAGS.num_epochs, FLAGS.capacity, FLAGS.num_threads,
                                              FLAGS.min_after_dequeue)
    labels = image_classes(os.path.join(FLAGS.dataset_dir, 'train_val'))
    net = model(batch_images, keep_prob=1.0, base_trainable=False,
                is_training=False)

    logits = net.output_logits(max(labels), scope='output_logits')
    predictions = tf.argmax(tf.nn.softmax(logits), 1)

    labels = tf.argmax(batch_labels, 1)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': tf.metrics.accuracy(predictions=predictions, labels=labels),
    })
    for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    latest_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir)

    tf.logging.info('Evaluating %s' % latest_ckpt_path)
    if FLAGS.options == 'val_loop':
        slim.evaluation.evaluation_loop(master='',
                                        session_config=sess_config,
                                        checkpoint_dir=FLAGS.ckpt_dir,
                                        logdir=os.path.join(FLAGS.ckpt_dir, 'summary', 'val'),
                                        num_evals=20,
                                        eval_op=list(names_to_updates.values()),
                                        timeout=10*60
                                        )
    else:
        slim.evaluation.evaluate_once(
            master='',
            logdir=os.path.join(FLAGS.ckpt_dir, 'summary', 'val'),
            session_config=sess_config,
            checkpoint_path=latest_ckpt_path,
            num_evals=100,
            eval_op=list(names_to_updates.values())
        )


def main():
    models = {
        'InceptRes': InceptResV2,
        "InceptV4": InceptV4
    }
    gpu_memory_fraction = FLAGS.gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    if FLAGS.options == 'train':
        _train(models[FLAGS.model], sess_config)
    else:
        _eval(models[FLAGS.model], sess_config)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_dir',
        type=str,
    )
    parser.add_argument(
        'ckpt_dir',
        type=str,
    )
    parser.add_argument(
        '--options',
        type=str,
        default='train'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='InceptRes'
    )
    parser.add_argument(
        '--base_trainable',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default=None
    )
    parser.add_argument(
        '--summary_name',
        type=str,
        default='train'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=299
    )
    parser.add_argument(
        '--resize_method',
        type=str,
        default='bicubic',
        help="thumbnail, random_crop, bicubic"
    )
    parser.add_argument(
        '--norm_method',
        type=str,
        default='no_change',
        help="minus_one_to_pos_one, standardize, default"
    )

    parser.add_argument(
        '--num_steps',
        type=int,
        default=50000
    )
    parser.add_argument(
        '--random_flip',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--distort_color',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--decay_frequency",
        type=int,
        default=3000
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.96
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=60
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min_after_dequeue",
        type=int,
        default=3000
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=None
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='SGD'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
