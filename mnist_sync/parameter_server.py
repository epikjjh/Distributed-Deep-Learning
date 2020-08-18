from mpi4py import MPI
import numpy as np
import tensorflow as tf
import time,sys

class ParameterServer:
    def __init__(self, params):
        self.var_size = params["size"]
        self.var_shape = params["shape"]
        self.total_batch = params["total_batch"]

        # Data for worker
        # For worker1
        self.w1_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

        # For worker2
        self.w2_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]

        # For worker1 + worker2 
        self.w_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.var_size)]
        self.ph_bucket = [tf.compat.v1.placeholder(shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]
    
        # TF variables
        with tf.compat.v1.variable_scope("ParameterServer", reuse=tf.compat.v1.AUTO_REUSE):
            self.var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]

        # Optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Apply gradients
        # Tuple: (gradient, variable)
        # Pack gradeint values 
        self.grads_and_vars = [(self.ph_bucket[i], self.var_bucket[i]) for i in range(self.var_size)]
        self.sync_gradients = self.optimizer.apply_gradients(self.grads_and_vars)
            
        # Create session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def update(self):
        for i in range(self.var_size):
            self.w_bucket[i] = self.w1_bucket[i] + self.w2_bucket[i]

        self.sess.run(self.sync_gradients, feed_dict={self.ph_bucket[i]:self.w_bucket[i] for i in range(self.var_size)})


if __name__ == "__main__":
    epoch = 1
    batch_size = 100
    comm = MPI.COMM_WORLD

    # Receive parameters from worker
    params = comm.recv(source=1, tag=0)

    ps = ParameterServer(params)
    # For broadcasting 
    bucket = [np.empty(ps.var_shape[i], dtype=np.float32) for i in range(ps.var_size)]

    for step in range(epoch):
        batch_num = int(ps.total_batch/batch_size)
        for batch_cnt in range(batch_num):
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
