from mnist import Model
from mpi4py import MPI
from typing import List
import numpy as np
import tensorflow as tf
import time

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
    
        # TF variables
        self.var_bucket = [tf.get_variable("v{}".format(i), shape=self.var_shape[i], initializer=tf.random_normal_initializer(), dtype=tf.float32) for i in range(self.var_size)]

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-4)

        # Apply gradients
        self.grads_and_vars = [(self.w1_bucket[i]+self.w2_bucket[i], self.var_bucket[i]) for i in range(self.var_size)]
        self.sync_gradients = self.optimizer.apply_gradients(self.grads_and_vars)
        
        # Create session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        self.sess.run(self.sync_gradients)


if __name__ == "__main__":
    epoch = 1
    batch_size = 100
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    
    # Worker1
    if rank == 1:
        w1 = SyncWorker(batch_size) 
        for step in range(epoch):
            grads_w1 = w1.work()
            
            # Send worker 1's grads
            for i in range(w1.var_size):
                comm.Send([grads_w1[i], MPI.DOUBLE], dest=0, tag=i+1)

    # Worker2
    elif rank == 2:
        w2 = SyncWorker(batch_size) 
        for step in range(epoch):
            grads_w2 = w2.work()
            
            # Send worker 1's grads
            for i in range(w2.var_size):
                comm.Send([grads_w2[i], MPI.DOUBLE], dest=0, tag=i+1)

    # Parameter Server
    else:
        ps = ParameterServer()
        for step in range(epoch):
            # Receive data from workers
            # From worker1
            for i in range(ps.var_size):
                comm.Recv([ps.w1_bucket[i], MPI.DOUBLE], source=1, tag=i+1)

            # From worker2
            for i in range(ps.var_size):
                comm.Recv([ps.w2_bucket[i], MPI.DOUBLE], source=2, tag=i+1)

            # Synchronize
            ps.update()


    '''
    ps = ParameterServer(comm, rank) 
    w1 = SyncWorker(comm, rank, batch_size)
    w2 = SyncWorker(comm, rank, batch_size) 

    conv1_gw = np.empty((5,5,1,32), dtype=np.float32)
    conv1_gb = np.empty(32, dtype=np.float32)
    conv2_gw = np.empty((5,5,32,64), dtype=np.float32)
    conv2_gb = np.empty(64, dtype=np.float32)
    fc1_gw = np.empty((7*7*64,1024), dtype=np.float32)
    fc1_gb = np.empty(1024, dtype=np.float32)
    fc2_gw = np.empty((1024,10), dtype=np.float32)
    fc2_gb = np.empty(10, dtype=np.float32)


    # Mapping variables
    vars = [
        conv1_gw, conv1_gb, conv2_gw, conv2_gb,
        fc1_gw, fc1_gb, fc2_gw, fc2_gb
    ]
        
    # Measure time
    if rank != 0:
        start = time.clock()

    for step in range(training_time):
        # Parameter Server
        if rank == 0:
            vars = ps.sync()

        # Worker1 
        elif rank == 1:
            grads_w1 = w1.work()

            # Send worker 1's grads
            for i in range(8):
                comm.Send([grads_w1[i], MPI.DOUBLE], dest=0, tag=i+1)
        
        # Worker2 
        else:
            grads_w2 = w2.work()

            # Send worker 2's grads
            for i in range(8):
                comm.Send([grads_w2[i], MPI.DOUBLE], dest=0, tag=i+1)


        # Receive data from parameter server
        comm.Bcast([vars[0], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[1], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[2], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[3], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[4], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[5], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[6], MPI.DOUBLE], root=0) 
        comm.Bcast([vars[7], MPI.DOUBLE], root=0) 
        
        # Update variables (worker)
        if rank == 1:
            w1.update(vars)

        elif rank == 2:
            w2.update(vars)


    # Evaluate model
    if self.rank != 0:
        end = time.clock()
        print(w1.sess.run(self.accuracy, feed_dict={w1.x: w1.test_x, w1.y_: w1.test_y_, w1.keep_prob: 1.0}))
        print(w2.sess.run(self.accuracy, feed_dict={w2.x: w2.test_x, w2.y_: w2.test_y_, w2.keep_prob: 1.0}))
        print(end-start)
    '''
