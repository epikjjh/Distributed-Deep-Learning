import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1

class Model:
    def __init__(self):
        # Data: mnist dataset
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.w = tf.Variable(tf.zeros([784, 10]), name='w')
        self.b = tf.Variable(tf.zeros([10]), name='b')
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        # Learning rate: 0.01
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)        
        self.train_step = self.optimizer.minimize(self.cost)

        # Gradients
        self.grads = self.optimizer.compute_gradients(self.cost, [self.w, self.b])

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
