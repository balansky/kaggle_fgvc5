from cores import InceptResV2, tf


class FurnitureRes(InceptResV2):

    def __init__(self, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(FurnitureRes, self).__init__(keep_prob, base_trainable, is_training, weight_decay,
                                           batch_norm_decay, batch_norm_epsilon)
        self.prelogits_names.append("InceptionResnetV2/output_logits")
        self._logit_scope = "output_logits"

    def output_logits(self, inputs, num_labels, reuse=tf.AUTO_REUSE):
        net, end_points = self.build_net(inputs, reuse)
        logits = self._logits(net, num_labels, "Logits")
        return logits

    def net_loss(self, batch_inputs, batch_labels, num_labels, reuse=tf.AUTO_REUSE, reg_scope=None):
        logits = self.output_logits(batch_inputs, num_labels, reuse=reuse)
        cross_entropy = tf.losses.softmax_cross_entropy(batch_labels, logits, scope='cross_entropy')
        res_loss = tf.reduce_mean(cross_entropy, name='loss')

        total_loss = tf.add_n([res_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope))

        return total_loss, logits


