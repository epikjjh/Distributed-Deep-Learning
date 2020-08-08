from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Model:
    def __init__(self):
        # Data: mnist dataset
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # CNN model
        with tf.variable_scope("mnist", reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None, 784])
            self.x_image = tf.reshape(self.x, [-1,28,28,1])
            self.y_ = tf.placeholder(tf.float32, [None, 10])
            '''First Conv layer'''
            # shape: [5,5,1,32]
            # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
            self.w_conv1 = tf.get_variable("v0", shape=[5,5,1,32], dtype = tf.float32)
            # shape: [32]
            self.b_conv1 = tf.get_variable("v1", shape=[32], dtype=tf.float32)
            # conv layer
            self.conv1 = tf.nn.conv2d(self.x_image, self.w_conv1, strides=[1,1,1,1], padding='SAME')
            # activation layer
            self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)
            self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            '''Second Conv layer'''
            # shape: [5,5,32,64]
            self.w_conv2 = tf.get_variable("v2", shape=[5,5,32,64], dtype=tf.float32)
            # shape: [64]
            self.b_conv2 = tf.get_variable("v3", shape=[64], dtype=tf.float32)
            # conv layer
            self.conv2 = tf.nn.conv2d(self.h_pool1, self.w_conv2, strides=[1,1,1,1], padding='SAME')
            # activation layer
            self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)
            self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            '''FC layer'''
            self.w_fc1 = tf.get_variable("v4", shape=[7*7*64, 1024], dtype=tf.float32)
            self.b_fc1 = tf.get_variable("v5", shape=[1024], dtype=tf.float32)
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)

            '''Dropout'''
            self.keep_prob = tf.placeholder(tf.float32)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

            '''Softmax layer'''
            self.w_fc2 = tf.get_variable("v6", shape=[1024, 10], dtype=tf.float32)
            self.b_fc2 = tf.get_variable("v7", shape=[10], dtype=tf.float32)
            self.y = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2)

            '''Cost function & optimizer'''
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
            self.optimizer = tf.train.AdamOptimizer(1e-4)
            self.train_step = self.optimizer.minimize(self.cost)
            
            self.var_shape =[]
            self.tmp_var = tf.trainable_variables()
            self.var_size = len(self.tmp_var)
            for i in range(self.var_size):
                self.var_shape.append(self.tmp_var[i].shape)
            self.layer = [tf.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]
            
            # Gradients
            self.grads = self.optimizer.compute_gradients(self.cost, self.layer)
            self.grads_and_vars = self.optimizer.apply_gradients(self.grads)
            #self.vars = [np.ones(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
            self.vars = [tf.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]
            # Data1
            self.d1 = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
    
            # Data2
            self.d2 = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
            
            # For evaluating
            self.prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
            self.test_x = self.data.test.images
            self.test_y_ = self.data.test.labels

            # Create session
            self.sess = tf.Session()

            # Initialize variables
            self.sess.run(tf.global_variables_initializer())
