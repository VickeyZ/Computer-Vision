# python3 run.py > run.log
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np
import sys

datasetDir = '../dataset/'
model = '../model/lenet'
modelDir = '../model/'

epochs = 20
batchSize = 128
rate = 0.001

mu = 0
sigma = 0.1

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, config.batchSize):
        batch_x, batch_y = x_data[offset:offset +
                                  config.batchSize], y_data[offset:offset+config.batchSize]
        accuracy = sess.run(accuracy_operation, feed_dict={
                            x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def lenet(x):
    # C1
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 6), mean=config.mu, stddev=config.sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[
                         1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # S2
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # C3
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=config.mu, stddev=config.sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[
                         1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # S4
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # C5
    fc0 = flatten(conv2)
    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(400, 120), mean=config.mu, stddev=config.sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    # F6
    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(120, 84), mean=config.mu, stddev=config.sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # output
    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(84, 10), mean=config.mu, stddev=config.sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits




if __name__ == '__main__':
    mnist = input_data.read_data_sets(config.datasetDir, reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    print("Image shape: {}".format(x_train[0].shape))
    print("Training set length: {}".format(len(x_train)))
    print("Validation set length: {}".format(len(x_validation)))
    print("Test set length: {}".format(len(x_test)))

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_validation = np.pad(
        x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    print()
    print("Image shape padded: {}".format(x_train[0].shape))

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 10)

    logits = lenet.lenet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Start training", file=sys.stderr)
        sess.run(tf.global_variables_initializer())
        num_examples = len(x_train)

        for i in range(config.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, num_examples, config.batchSize):
                end = offset + config.batchSize
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={
                         x: batch_x, y: batch_y})

            validation_accuracy = evaluate(x_validation, y_validation)
            print(i, validation_accuracy)
            print(i, validation_accuracy, file=sys.stderr)

        saver.save(sess, config.model)
        print("Finish training", file=sys.stderr)
        print()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(config.modelDir))
        test_accuracy = evaluate(x_test, y_test)
        print("Accuracy with {} test data = {:.3f}".format(
            len(x_test), test_accuracy))
        print("Accuracy with {} test data = {:.3f}".format(
            len(x_test), test_accuracy), file=sys.stderr)
