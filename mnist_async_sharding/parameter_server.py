from mpi4py import MPI
import numpy as np
import tensorflow as tf
import time,sys

class ParameterServer:
    def __init__(self, params, rank, num_ps, num_workers):
        # Set rank of parameter server
        # rank: 0 ~ the number of parameter servers -1
        self.rank = rank
        
        # Set number of parameter servers & workers
        self.num_ps = num_ps
        self.num_workers = num_workers
        
        '''
            Parameter length:
                Incase of last parameter server:
                    Total length of parameters // number of parameter servers + Total length of parameters % number of parameter servers

                else:
                    Total length of parameters // number of parameter servers

            Parameter Server rank i: total_parameter[parameter_legnth*i:parameter_length*(i+1)]
        '''
        # Total length of parameters
        self.total_var_size = params["size"]

        # set local length of parameters
        self.local_var_size = self.total_var_size // num_ps
        if rank == num_ps-1:
            self.local_var_size += self.total_var_size % num_ps

        self.var_shape = params["shape"]
        self.total_batch = params["total_batch"]

        # For applying gradients
        if rank != num_ps-1:
            self.w_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.local_var_size * rank, self.local_var_size * (rank+1))]
            self.ph_bucket = [tf.compat.v1.placeholder(shape=self.var_shape[i], dtype=tf.float32) for i in range(self.local_var_size * rank, self.local_var_size * (rank+1))]
        else:
            self.w_bucket = [np.empty(self.var_shape[i], dtype=np.float32) for i in range(self.total_var_size - self.local_var_size, self.total_var_size)]
            self.ph_bucket = [tf.compat.v1.placeholder(shape=self.var_shape[i], dtype=tf.float32) for i in range(self.total_var_size - self.local_var_size, self.total_var_size)]
    
        # TF variables
        with tf.compat.v1.variable_scope("ParameterServer", reuse=tf.compat.v1.AUTO_REUSE):
            if rank != num_ps-1:
                self.var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.local_var_size * rank, self.local_var_size * (rank+1))]
            else:
                self.var_bucket = [tf.compat.v1.get_variable("v{}".format(i), shape=self.var_shape[i], dtype=tf.float32) for i in range(self.total_var_size - self.local_var_size, self.total_var_size)]

        # Optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Apply gradients
        # Tuple: (gradient, variable)
        # Pack gradeint values
        self.grads_and_vars = [(self.ph_bucket[i], self.var_bucket[i]) for i in range(self.local_var_size)]
        self.sync_gradients = self.optimizer.apply_gradients(self.grads_and_vars)
            
        # Create session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # Synchronize
    def update(self):
        self.sess.run(self.sync_gradients, feed_dict={self.ph_bucket[i]:self.w_bucket[i] for i in range(self.local_var_size)})


if __name__ == "__main__":
    epoch = 1
    batch_size = 100
    comm = MPI.COMM_WORLD
    info = MPI.Status()

    # Set rank of parameter server
    # rank: 0 ~ number of parameter servers-1
    rank = comm.Get_rank()
    
    # Set number of parameter servers & workers
    num_ps = int(sys.argv[2])
    num_workers = comm.Get_size() - num_ps

    # Receive parameters from worker
    params = comm.recv(source=num_ps, tag=0)
    ps = ParameterServer(params, rank, num_ps, num_workers)

    # For sending
    if rank != num_ps-1:
        bucket = [np.empty(ps.var_shape[i], dtype=np.float32) for i in range(ps.local_var_size*ps.rank, ps.local_var_size*(ps.rank+1))]
    else:
        bucket = [np.empty(ps.var_shape[i], dtype=np.float32) for i in range(ps.total_var_size-ps.local_var_size, ps.total_var_size)]

    for step in range(epoch):
        # Batch iteration number: batch count * number of worker
        batch_num = int(ps.total_batch/batch_size)*num_workers
        for batch_cnt in range(batch_num):
            # Receive data from worker
            for i in range(ps.local_var_size):
                comm.Recv([ps.w_bucket[i], MPI.FLOAT], source=MPI.ANY_SOURCE, tag=i, status=info)
                
            # Synchronize
            ps.update()

            # Prepare sending values
            for i in range(ps.local_var_size):
                bucket[i] = ps.var_bucket[i].eval(session=ps.sess)

            # send to worker
            for i in range(ps.local_var_size):
                comm.Send([bucket[i], MPI.FLOAT], dest=info.Get_source(), tag=i)
