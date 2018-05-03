import argparse
from cores.models import *
from cores.dataset import *
from cores.utils.image import Decoder
from cores.utils.ops import lr_decay_op, mutli_gpu_train_op
from generate_labels import generate_child_labels
import os
# from cores.utils.plt import plot_confusion_matrix
# import numpy as np
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_confusion_matrix
# import pylab as pl
from pandas_ml import ConfusionMatrix

FLAGS = None
tf.logging.set_verbosity(20)


# def _train(model, sess_config):
#
#     img_decoder = Decoder(FLAGS.input_size, FLAGS.resize_method, FLAGS.norm_method, random_flip=FLAGS.random_flip,
#                           distort_color=FLAGS.distort_color)
#     net = model('train', FLAGS.dataset_dir, FLAGS.batch_size, img_decoder,
#                 FLAGS.num_epochs, FLAGS.capacity, FLAGS.num_threads,
#                 FLAGS.min_after_dequeue, num_gpus=FLAGS.num_gpus,
#                 keep_prob=FLAGS.keep_prob, base_trainable=FLAGS.base_trainable,
#                 is_training=True)
#
#     total_loss = net.net_loss()
#
#     train_step, global_step = train_op(total_loss, FLAGS.learning_rate, FLAGS.optimizer, FLAGS.decay_frequency,
#                                        FLAGS.decay_rate)
#
#     saver = tf.train.Saver(max_to_keep=5)
#     summary_op = tf.summary.merge_all()
#     summary_dir = os.path.join(FLAGS.ckpt_dir, 'summary', FLAGS.summary_name)
#     summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
#     slim.learning.train(train_step, FLAGS.ckpt_dir,
#                         session_config=sess_config,
#                         init_fn=net.restore_fn(FLAGS.ckpt_dir, FLAGS.pretrained_ckpt),
#                         log_every_n_steps=10,
#                         global_step=global_step, number_of_steps=FLAGS.num_steps, summary_op=summary_op,
#                         summary_writer=summary_writer, saver=saver, save_interval_secs=60, save_summaries_secs=60)


def _train(net, train_set, sess_config):

    if not os.path.exists(os.path.join(FLAGS.dataset_dir, 'child_labels.json')):
        generate_child_labels(FLAGS.dataset_dir, [], [])

    with tf.device("/device:CPU:0"):
        global_step = tf.train.get_or_create_global_step()
        lr = lr_decay_op(FLAGS.decay_frequency, FLAGS.decay_rate)(FLAGS.learning_rate, global_step)
        optimizer = tf.train.GradientDescentOptimizer(lr)

    clone_grads = []
    total_losses = []
    with slim.arg_scope([slim.model_variable, slim.variable],
                        device='/device:CPU:0'):
        for i in range(0, FLAGS.num_gpus):
            clone_scope = "clone_%d" % i
            with tf.name_scope(clone_scope):
                inputs = train_set.batch_inputs()
                with tf.device('/device:GPU:%d' % i):
                    with tf.variable_scope(tf.get_variable_scope()):

                        net_loss = net.net_loss(inputs, reuse=True if i > 0 else None, reg_scope=clone_scope)

                        scale_loss = tf.div(net_loss, 1.0*FLAGS.num_gpus, name='scale_loss_%d' % i)
                        trainable_variable = tf.trainable_variables()
                        clone_grad = optimizer.compute_gradients(scale_loss, trainable_variable)
                        total_losses.append(scale_loss)
                        clone_grads.append(clone_grad)

    train_step = mutli_gpu_train_op(optimizer, global_step, clone_grads, total_losses)

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


def _eval(net, data_set, sess_config):

    inputs = data_set.batch_inputs()
    names_to_values, names_to_updates = net.net_eval(inputs)

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
            num_evals=1000,
            eval_op=list(names_to_updates.values())
        )


def _confusion_matrix(net, data_set, sess_config):
    preds = []
    lbs = []
    count = 0
    orig_to_child, child_to_orig = load_child_labels(FLAGS.dataset_dir)
    with tf.Session(config=sess_config) as sess:
        inputs = data_set.batch_inputs()
        label_inputs = inputs[1]
        output_logits = net.output_logits(inputs[0])
        sess.run(tf.initialize_local_variables())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        if len(inputs) == 3:
            output_logits = output_logits[-1]
        output_softmax = tf.nn.softmax(output_logits)
        prediction = tf.argmax(output_softmax, axis=1)
        labels = tf.argmax(label_inputs, axis=1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while True:
            try:
                count += 1
                print("%s Evalutions..." % (count * FLAGS.batch_size))
                pred_values, label_values = sess.run([prediction, labels])
                preds.extend([child_to_orig[pd] for pd in pred_values.tolist()])
                lbs.extend([child_to_orig[lv] for lv in label_values.tolist()])
            except Exception:
                coord.request_stop()
                coord.join(threads)
                break

    cm = ConfusionMatrix(lbs, preds)
    cm.print_stats()
    cm.plot()
    plt.show()
    # np.set_printoptions(precision=2)
    # plt.figure()
    # fig, ax = plot_confusion_matrix(confuse_value, figsize=(100, 100))

    # plt.show()







def main():
    models = {
        'InceptRes': (FurnitureInceptRes, DataSet),
        'InceptResMixed': (FurnitureInceptResMixed, DataSetWithParent),
        'InceptResParent': (FurnitureInceptResParent, DataSetWithParent),
    }
    options = {
        'train': _train,
        'val': _eval,
        'val_loop': _eval,
        'confuse_matrix': _confusion_matrix
    }
    gpu_memory_fraction = FLAGS.gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    model, dataset = models[FLAGS.model]
    if FLAGS.options == 'train':
        img_decoder = Decoder(FLAGS.input_size, FLAGS.resize_method, FLAGS.norm_method, random_flip=FLAGS.random_flip,
                              distort_color=FLAGS.distort_color)
        data_set = dataset('train', FLAGS.dataset_dir, FLAGS.batch_size, img_decoder,
                            FLAGS.num_epochs, FLAGS.capacity, FLAGS.num_threads,
                            FLAGS.min_after_dequeue, num_gpus=FLAGS.num_gpus,)
        net = model(FLAGS.dataset_dir, keep_prob=FLAGS.keep_prob, base_trainable=FLAGS.base_trainable,
                    is_training=True)
    else:
        img_decoder = Decoder(FLAGS.input_size, FLAGS.resize_method, FLAGS.norm_method)
        data_set = dataset('validation', FLAGS.dataset_dir, FLAGS.batch_size, img_decoder,
                          FLAGS.num_epochs, FLAGS.capacity, FLAGS.num_threads,
                          FLAGS.min_after_dequeue, num_gpus=FLAGS.num_gpus,)
        net = model(FLAGS.dataset_dir)
    options[FLAGS.options](net, data_set, sess_config)


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
        default='minus_one_to_pos_one',
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
        "--num_gpus",
        type=int,
        default=1
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='SGD'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
