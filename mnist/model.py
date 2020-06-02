import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class Model:
    def __init__(self):
        # Data: mnist dataset
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # CNN model
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.x_image = tf.reshape(self.x, [-1,28,28,1])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        '''First Conv layer'''
        # shape: [5,5,1,32]
        self.w_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1)) 
        # shape: [32]
        self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        # conv layer
        self.conv1 = tf.nn.conv2d(self.x_image, self.w_conv1, strides=[1,1,1,1], padding='SAME')
        # activation layer
        self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)
        self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        '''Second Conv layer'''
        # shape: [5,5,32,64] 
        self.w_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
        # shape: [64] 
        self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        # conv layer
        self.conv2 = tf.nn.conv2d(self.h_pool1, self.w_conv2, strides=[1,1,1,1], padding='SAME')
        # activation layer
        self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)
        self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        '''FC layer'''
        self.w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
        self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)

        '''Dropout'''
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        '''Softmax layer'''
        self.w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2)

        '''Cost function & optimizer'''
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_step = self.optimizer.minimize(self.cost)

        # Gradients
        self.grads = self.optimizer.compute_gradients(self.cost, 
                        [self.w_conv1, self.b_conv1,
                        self.w_conv2, self.b_conv2,
                        self.w_fc1, self.b_fc1,
                        self.w_fc2, self.b_fc2])
        # Data1
        self.d1_conv1_gw = np.empty((5,5,1,32), dtype=np.float32) 
        self.d1_conv1_gb = np.empty(32, dtype=np.float32) 
        self.d1_conv2_gw = np.empty((5,5,32,64), dtype=np.float32) 
        self.d1_conv2_gb = np.empty(64, dtype=np.float32) 
        self.d1_fc1_gw = np.empty((7*7*64,1024), dtype=np.float32) 
        self.d1_fc1_gb = np.empty(1024, dtype=np.float32) 
        self.d1_fc2_gw = np.empty((1024,10), dtype=np.float32) 
        self.d1_fc2_gb = np.empty(10, dtype=np.float32) 
        # Data2   
        self.d2_conv1_gw = np.empty((5,5,1,32), dtype=np.float32) 
        self.d2_conv1_gb = np.empty(32, dtype=np.float32) 
        self.d2_conv2_gw = np.empty((5,5,32,64), dtype=np.float32) 
        self.d2_conv2_gb = np.empty(64, dtype=np.float32) 
        self.d2_fc1_gw = np.empty((7*7*64,1024), dtype=np.float32) 
        self.d2_fc1_gb = np.empty(1024, dtype=np.float32) 
        self.d2_fc2_gw = np.empty((1024,10), dtype=np.float32) 
        self.d2_fc2_gb = np.empty(10, dtype=np.float32) 
        # Broadcasting values
        self.bcast_conv1_w = np.empty((5,5,1,32), dtype=np.float32) 
        self.bcast_conv1_b = np.empty(32, dtype=np.float32) 
        self.bcast_conv2_w = np.empty((5,5,32,64), dtype=np.float32) 
        self.bcast_conv2_b = np.empty(64, dtype=np.float32) 
        self.bcast_fc1_w = np.empty((7*7*64,1024), dtype=np.float32) 
        self.bcast_fc1_b = np.empty(1024, dtype=np.float32) 
        self.bcast_fc2_w = np.empty((1024,10), dtype=np.float32) 
        self.bcast_fc2_b = np.empty(10, dtype=np.float32) 

        # For evaluating
        self.prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
        self.test_x = self.data.test.images
        self.test_y_ = self.data.test.labels 

        # Create session
        self.sess = tf.Session()

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())


    def train(self):
        # Train specific model here
        raise NotImplementedError("Train method not implemented")
