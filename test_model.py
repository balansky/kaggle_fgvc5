from cores.utils.data import batch_inputs
from cores.utils.image import Decoder
from cores.models import FurnitureAttentionTrainer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def test_inputs():
    img_decoder = Decoder(299, 'default', 'no_change')

    with tf.Session() as sess:
        batch_images, batch_labels = batch_inputs('train', 'dataset', 16, img_decoder)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # for _ in range(10):

            # images, ages, genders = sess.run([batch_images, batch_ages, batch_genders])
            # print(ages)

        images, labels = sess.run([batch_images, batch_labels])
        coord.request_stop()
        coord.join(threads)

    fig = plt.figure()
    print(images[0])
    for i in range(len(images)):
        fig.add_subplot(6, 6, i+1)
        plt.axis('off')
        plt.title("label: %s" % (np.argmax(labels[i]), ))
        plt.imshow(np.clip(images[i], 0, 1))
    plt.show()

def test_model():
    img_decoder = Decoder(299, 'default', 'no_change')

    with tf.Session() as sess:
        model = FurnitureAttentionTrainer('train', 'dataset', 32, img_decoder)
        logits = model.net_loss()

# test_inputs()
test_model()