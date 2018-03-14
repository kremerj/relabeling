import tensorflow as tf
import numpy as np


class AlexNet(object):
    def __init__(self, seed=66478, width=227, height=227, num_channels=3, classes=2):
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.classes = classes
        self.checkpoint_file = 'cache/alexnet-initial'
        self.model_weights_file = "models/bvlc_alexnet.npy"
        self.seed = seed
        self._create_variables()

    def _create_variables(self):
        net_data = np.load(self.model_weights_file).item()
        self.conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
        self.conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
        self.conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
        self.conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
        self.conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
        self.conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
        self.conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
        self.conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
        self.conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
        self.conv5b = tf.Variable(net_data["conv5"][1], trainable=False)
        self.fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
        self.fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
        self.fc7W = tf.Variable(net_data["fc7"][0], trainable=False)
        self.fc7b = tf.Variable(net_data["fc7"][1], trainable=False)
        self.fc8W = tf.Variable(tf.truncated_normal([4096, self.classes], stddev=0.01, seed=self.seed, dtype=tf.float32))
        self.fc8b = tf.Variable(tf.constant(0.0, shape=[self.classes], dtype=tf.float32))
        self.weights = [self.fc8W]

    def logit(self, input_data):
        def conv(input_data, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = input_data.get_shape()[-1]
            assert c_i % group == 0
            assert c_o % group == 0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

            if group == 1:
                conv = convolve(input_data, kernel)
            else:
                input_groups = tf.split(input_data, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        conv1 = tf.nn.relu(conv(input_data, self.conv1W, self.conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.relu(conv(maxpool1, self.conv2W, self.conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.relu(conv(maxpool2, self.conv3W, self.conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))
        conv4 = tf.nn.relu(conv(conv3, self.conv4W, self.conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))
        conv5 = tf.nn.relu(conv(conv4, self.conv5W, self.conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), self.fc6W, self.fc6b)
        fc7 = tf.nn.relu_layer(fc6, self.fc7W, self.fc7b)
        fc8 = tf.nn.xw_plus_b(fc7, self.fc8W, self.fc8b)

        return fc8
