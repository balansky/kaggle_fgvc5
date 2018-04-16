import tensorflow as tf


def lr_decay_op(decay_frequency=None, decay_rate=None):
    def _lr_decay_fn(init_learning_rate, gs):

        if decay_frequency and decay_rate:
            return tf.train.exponential_decay(init_learning_rate, gs, decay_frequency,
                                              decay_rate, staircase=True)
        else:
            return None
    return _lr_decay_fn

def softmax_accuracy_op(logits, labels, name='accuracy'):
    res_softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(res_softmax, 1)
    label_idx = tf.argmax(labels, 1)
    correct_prediction = tf.equal(predictions, label_idx)
    res_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=name)
    tf.summary.scalar(name, res_accuracy)
    return res_accuracy


def train_op(total_loss, learning_rate, optimizer, decay_frequency=None, decay_rate=None, clip_gradients=None):

    _lr_decay_fn = lr_decay_op(decay_frequency, decay_rate)

    global_step = tf.train.get_or_create_global_step()

    train_step = tf.contrib.layers.optimize_loss(
        loss=total_loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clip_gradients=clip_gradients,
        learning_rate_decay_fn=_lr_decay_fn)

    for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

    return train_step, global_step


def mutli_gpu_train_op(optimizer, global_step, clone_grads, loss, clone_scope='clone_0'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=clone_scope)
    avg_grads = []
    for grad_and_vars in zip(*clone_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(tf.expand_dims(g, 0))
        if grads:
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0, name=var.op.name + '/average_grads')
            avg_grads.append((grad, var))
    grad_updates = optimizer.apply_gradients(avg_grads, global_step=global_step)

    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_step = tf.identity(loss, name='train_op')

    return train_step