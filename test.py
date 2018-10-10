"""Builds the deep network."""

import tensorflow as tf
import numpy as np
import time
import os

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = 2
NUM_FEATURES = 3
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5000
DROP_OUT = 0.8
MAX_STEPS = 100000

def generate_train_data():
    x = np.random.rand(1000000, 4)
    x[:,0] = x[:,1] * x[:,1] + x[:,2] * x[:,2] + x[:,3] * x[:,3]
    x[:,0] = x[:,0] > 1
    x.tofile('train_data_file.bin')

def inference(input_dim, x):
    """Build the deep model. """
    # dense layer 1
    with tf.variable_scope('dense1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[input_dim, 64],
                                  initializer=tf.truncated_normal_initializer(stddev=4e-2, dtype=tf.float32),
                                  dtype=tf.float32)

        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))
        dense1 = tf.nn.relu(tf.matmul(x, weights) + biases, name=scope.name)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    drop = tf.nn.dropout(dense1, keep_prob)

    # dense layer2
    with tf.variable_scope('dense2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[64, 32],
                                  initializer=tf.truncated_normal_initializer(stddev=4e-2, dtype=tf.float32),
                                  dtype=tf.float32)

        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.1))
        dense2 = tf.nn.relu(tf.matmul(drop, weights) + biases, name=scope.name)

    # linear layer(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[32, NUM_CLASSES],
                                  initializer=tf.truncated_normal_initializer(stddev=4e-2, dtype=tf.float32),
                                  dtype=tf.float32)

        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        score = tf.add(tf.matmul(dense2, weights), biases, name=scope.name)

    return score, keep_prob


def train(batch_size=32):
    """Train deep model."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_data = np.fromfile('train_data_file.bin', dtype=np.float)
        train_data = np.reshape(train_data, [-1, 1 + NUM_FEATURES])
        input_features = tf.placeholder(dtype=tf.float32, shape=[None, NUM_FEATURES], name='features')
        input_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="labels")
        logits, keep_prob = inference(NUM_FEATURES, input_features)

        # Calculate cross entropy loss.
        labels = tf.cast(tf.logical_xor(tf.cast(input_labels, tf.bool), [True, False]), tf.int16)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Optimizer
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
        correct_prediction = tf.equal(input_labels,
                                      tf.cast(
                                          tf.greater(tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), [-1, 1]), axis=1),
                                                     [0]), tf.float32)
                                      )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            num_epochs = NUM_EPOCHS
            num_examples = train_data.shape[0]

            step = 0
            for epoch in range(num_epochs):
                # Generating next batch here
                np.random.shuffle(train_data)
                train_labels = train_data[:, 0]
                train_features = np.reshape(train_data[:, 1:], [-1, NUM_FEATURES])

                num_sub_step = int(num_examples / batch_size)
                for i in range(num_sub_step):
                    batch = []
                    batch.append(train_features[i * batch_size: (i + 1) * batch_size])
                    batch.append(np.reshape(train_labels[i * batch_size: (i + 1) * batch_size], (batch_size, 1)))

                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss],
                                             feed_dict={input_features: batch[0], input_labels: batch[1],
                                                        keep_prob: DROP_OUT})

                    duration = time.time() - start_time
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    step = step + 1

                    if step % 1000 == 0:
                        examples_per_sec = batch_size / duration

                        train_accuracy = accuracy.eval(
                            feed_dict={input_features: batch[0], input_labels: batch[1], keep_prob: 1.0})

                        format_str = ('Epoch %d: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                                      'sec/batch), training accuracy = %g')
                        print (format_str % (epoch, step, loss_value,
                                             examples_per_sec, duration, train_accuracy))

                    # Save model
                    if step % 100 == 0 or (step + 1) == MAX_STEPS:
                        checkpoint_path = os.path.join('model', 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

def predict(x):
    with tf.Graph().as_default():
        input_features = tf.placeholder(dtype=tf.float32, shape=[None, NUM_FEATURES], name="features")
        logits, keep_prob = inference(NUM_FEATURES, input_features)
        prediction = tf.cast(tf.greater(tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), [-1, 1]), axis=1), [0]),
                             tf.int32, name="prediction_bytes")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            ckpt = tf.train.get_checkpoint_state('model')
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            # Execute the prediction
            preds = sess.run(prediction, feed_dict={input_features: x, keep_prob: 1.0})
            print(preds)

            coord.request_stop()
            coord.join(threads)

def main(argv=None):
    # generate_train_data()
    #train()
    x = np.random.rand(10, 4)
    x[:,0] = x[:,1] * x[:,1] + x[:,2] * x[:,2] + x[:,3] * x[:,3]
    x[:,0] = x[:,0] > 1

    predict(x[:,1:])
    print(x[:, 0])

if __name__ == '__main__':
    tf.app.run()
