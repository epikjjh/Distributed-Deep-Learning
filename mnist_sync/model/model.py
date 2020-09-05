import tensorflow as tf
import pickle, gzip, urllib.request
import numpy as np
import pandas as pd

class Model:
    def __init__(self):
        # Data: mnist dataset
        urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, _, test_set = pickle.load(f, encoding='latin1')

        self.x_train, y_train = train_set
        self.x_test, y_test = test_set

        self.y_train = pd.get_dummies(y_train)
        self.y_test = pd.get_dummies(y_test)
        
        # CNN model
        with tf.compat.v1.variable_scope("mnist", reuse=tf.compat.v1.AUTO_REUSE):
            self.x = tf.compat.v1.placeholder(tf.float32, [None, 784])
            self.x_image = tf.reshape(self.x, [-1,28,28,1])
            self.y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

            '''First Conv layer'''
            # shape: [5,5,1,32]
            self.w_conv1 = tf.compat.v1.get_variable("v0", shape=[5,5,1,32], dtype=tf.float32) 
            # shape: [32]
            self.b_conv1 = tf.compat.v1.get_variable("v1", shape=[32], dtype=tf.float32)
            # conv layer
            self.conv1 = tf.nn.conv2d(self.x_image, self.w_conv1, strides=[1,1,1,1], padding='SAME')
            # activation layer
            self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)
            self.h_pool1 = tf.nn.max_pool2d(self.h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            '''Second Conv layer'''
            # shape: [5,5,32,64] 
            self.w_conv2 = tf.compat.v1.get_variable("v2", shape=[5,5,32,64], dtype=tf.float32)
            # shape: [64] 
            self.b_conv2 = tf.compat.v1.get_variable("v3", shape=[64], dtype=tf.float32)
            # conv layer
            self.conv2 = tf.nn.conv2d(self.h_pool1, self.w_conv2, strides=[1,1,1,1], padding='SAME')
            # activation layer
            self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)
            self.h_pool2 = tf.nn.max_pool2d(self.h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
            '''FC layer'''
            self.w_fc1 = tf.compat.v1.get_variable("v4", shape=[7*7*64, 1024], dtype=tf.float32)
            self.b_fc1 = tf.compat.v1.get_variable("v5", shape=[1024], dtype=tf.float32)
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)

            '''Dropout'''
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, rate=1.0-self.keep_prob)

            '''Softmax layer'''
            self.w_fc2 = tf.compat.v1.get_variable("v6", shape=[1024, 10], dtype=tf.float32)
            self.b_fc2 = tf.compat.v1.get_variable("v7", shape=[10], dtype=tf.float32)
            self.logits = tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2
            self.y = tf.nn.softmax(self.logits)

            '''Cost function & optimizer'''
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        
            # Variables
            self.var_bucket = tf.compat.v1.trainable_variables()
            self.var_size = len(self.var_bucket)
            self.var_shape = [var.shape for var in self.var_bucket]

            # Gradients
            self.grads = self.optimizer.compute_gradients(self.cost, self.var_bucket)

            # For evaluating
            self.prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
            self.train_step = self.optimizer.minimize(self.cost)
            
            # Create session
            self.sess = tf.compat.v1.Session()

            # Initialize variables
            self.sess.run(tf.compat.v1.global_variables_initializer())
