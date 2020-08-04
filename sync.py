from mnist import Model
from mpi4py import MPI
from typing import List
import numpy as np
import tensorflow as tf
import time,sys

class SyncWorker(Model):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def work(self) -> List:
        x_batch, y_batch = self.data.train.next_batch(self.batch_size)
        # Compute gradient values
        ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        # ret -> ((gw, w), (gb, b))
        grads = [grad for grad, var in ret]

        # Tuple: (gradient, variable)
        # Pack gradeint values 
        # Data: [gw_conv1, gb_conv1, gw_conv2, gb_conv2, gw_fc1, gb_fc1, gw_fc2, gb_fc2]
        # Send data to parameter server
        return grads


class ParameterServer:
    def __init__(self):
        self.var_size = 8
        self.var_shape = [
            [5,5,1,32],
            [32],
            [5,5,32,64],
            [64],
            [7*7*64, 1024],
            [1024],
            [1024, 10],
            [10]
        ]

        # Data for worker
        # For worker1
        self.w1_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

        # For worker2
        self.w2_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

        # For worker1 + worker2 
        self.w_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
    
        # TF variables
        with tf.compat.v1.variable_scope("ParameterServer", reuse=tf.compat.v1.AUTO_REUSE):
            self.var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]

        # Optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Apply gradients
        # Tuple: (gradient, variable)
        # Pack gradeint values 
        self.grads_and_vars = [(self.w_bucket[i], self.var_bucket[i]) for i in range(self.var_size)]
        self.sync_gradients = self.optimizer.apply_gradients(self.grads_and_vars)
            
        # Create session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def update(self):
        for i in range(self.var_size):
            self.w_bucket[i] = self.w1_bucket[i] + self.w2_bucket[i]

        self.grads_and_vars = [(self.w_bucket[i], self.var_bucket[i]) for i in range(self.var_size)]
        self.sess.run(self.sync_gradients)


if __name__ == "__main__":
    epoch = 10
    batch_size = 50
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    
    # Worker1
    if rank == 1:
        start = time.clock()

        w1 = SyncWorker(batch_size) 
        # For broadcasting 
        bucket = [np.empty(w1.var_shape[i], dtype=np.float32) for i in range(w1.var_size)]

        with tf.compat.v1.variable_scope("mnist", reuse=tf.compat.v1.AUTO_REUSE):
            var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=w1.var_shape[i], dtype=tf.float32) for i in range(w1.var_size)]
            bucket_assign = [tf.compat.v1.assign(var_bucket[i], bucket[i]) for i in range(w1.var_size)]

            for step in range(epoch):
                grads_w1 = w1.work()
                # Send worker 1's grads
                for i in range(w1.var_size):
                    comm.Send([grads_w1[i], MPI.FLOAT], dest=0, tag=i+1)

                # Receive data from parameter server
                for i in range(w1.var_size):
                    comm.Bcast([bucket[i], MPI.FLOAT], root=0) 
                # Assign broadcasted values

                # Problem section: duplicate tensor graph
                bucket_assign = [tf.compat.v1.assign(var_bucket[i], bucket[i]) for i in range(w1.var_size)]

                w1.sess.run(bucket_assign)
            
            end = time.clock()
            print("Worker{} Accuracy: {}".format(rank,w1.sess.run(w1.accuracy, feed_dict={w1.x: w1.test_x, w1.y_: w1.test_y_, w1.keep_prob: 1.0})))
            print("Time: {}".format(end-start))
     

    # Worker2
    elif rank == 2:
        start = time.clock()

        w2 = SyncWorker(batch_size) 
        # For broadcasting 
        bucket = [np.empty(w2.var_shape[i], dtype=np.float32) for i in range(w2.var_size)]

        with tf.compat.v1.variable_scope("mnist", reuse=tf.compat.v1.AUTO_REUSE):
            var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=w2.var_shape[i], dtype=tf.float32) for i in range(w2.var_size)]
            bucket_assign = [tf.compat.v1.assign(var_bucket[i], bucket[i]) for i in range(w2.var_size)]

            for step in range(epoch):
                grads_w2 = w2.work()
                
                # Send worker 1's grads
                for i in range(w2.var_size):
                    comm.Send([grads_w2[i], MPI.FLOAT], dest=0, tag=i+1)

                # Receive data from parameter server
                for i in range(w2.var_size):
                    comm.Bcast([bucket[i], MPI.FLOAT], root=0) 

                # Assign broadcasted values

                # Problem section: duplicate tensor graph
                bucket_assign = [tf.compat.v1.assign(var_bucket[i], bucket[i]) for i in range(w2.var_size)]

                w2.sess.run(bucket_assign)

            end = time.clock()
            print("Worker{} Accuracy: {}".format(rank,w2.sess.run(w2.accuracy, feed_dict={w2.x: w2.test_x, w2.y_: w2.test_y_, w2.keep_prob: 1.0})))
            print("Time: {}".format(end-start))

    # Parameter Server
    else:
        ps = ParameterServer()
        # For broadcasting 
        bucket = [np.empty(ps.var_shape[i], dtype=np.float32) for i in range(ps.var_size)]

        for step in range(epoch):
            # Receive data from workers
            # From worker1
            for i in range(ps.var_size):
                comm.Recv([ps.w1_bucket[i], MPI.FLOAT], source=1, tag=i+1)

            # From worker2
            for i in range(ps.var_size):
                comm.Recv([ps.w2_bucket[i], MPI.FLOAT], source=2, tag=i+1)

            # Synchronize
            ps.update()

            # Broadcast values
            for i in range(ps.var_size):
                bucket[i] = ps.var_bucket[i].eval(session=ps.sess)

            # send to worker 1, 2
            for i in range(ps.var_size):
                comm.Bcast([bucket[i], MPI.FLOAT], root=0) 
