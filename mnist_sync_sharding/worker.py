from model import Model
from mpi4py import MPI
from typing import List
import numpy as np
import tensorflow as tf
import time,sys

class SyncWorker(Model):
    def __init__(self, batch_size, rank, num_ps, num_workers):
        super().__init__()
        
        # Set rank of worker
        # rank: number of parameter servers ~ number of parameter servers + number of workers - 1
        self.rank = rank
        
        # Set number of parameter servers & workers
        self.num_workers = num_workers
        self.num_ps = num_ps
        self.avg_var_size = self.var_size // self.num_ps
        self.local_var_size = self.avg_var_size + self.var_size % self.num_ps

        self.batch_size = batch_size
        
        # MPI.Send input bucket
        self.grad_buckets = [tf.compat.v1.placeholder(shape=self.var_shape[i], dtype=tf.float32) for i in range(self.var_size)]
        
        # MPI.Send tensor
        self.senders = [tf.py_function(func=self.wrap_send(i), inp=[self.grad_buckets[i]], Tout=[]) for i in range(self.var_size)]

    def wrap_send(self, num):
        def send(grad):
            # Send data to parameter server
            ind = num // self.avg_var_size # parameter server's rank
            if num >= self.var_size - self.local_var_size:
                ind = self.num_ps-1
            comm.Send([grad, MPI.FLOAT], dest=ind, tag=num-(ind*self.avg_var_size))
            return None
        return send

    def work(self, cnt):
        x_batch = self.x_train[self.batch_size*cnt:self.batch_size*(cnt+1)]
        y_batch = self.y_train[self.batch_size*cnt:self.batch_size*(cnt+1)]
        # compute gradients
        ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        # gradient tuple
        grads = [grad for grad, var in ret]
        
        # Send gradients to each parameter server
        for i in range(self.var_size):
            self.sess.run([self.senders[i]], feed_dict={self.grad_buckets[i]: grads[i]})



if __name__ == "__main__":
    epoch = 1
    batch_size = 100
    comm = MPI.COMM_WORLD

    # Set rank of worker
    # rank: number of parameter servers ~ number of parameter servers + number of workers - 1
    rank = comm.Get_rank()
    
    # Set number of parameter servers & workers
    num_workers = int(sys.argv[2])
    num_ps = comm.Get_size() - num_workers
 
    start = time.clock()
    worker = SyncWorker(batch_size, rank, num_ps, num_workers)

    # Send parameters to all parameter servers
    if worker.rank == worker.num_ps:
        data = {"size": worker.var_size, "shape": worker.var_shape, "total_batch": worker.x_train.shape[0]}
        for i in range(worker.num_ps):
            comm.send(data, dest=i, tag=0)

    # For broadcasting
    bucket = [np.empty(worker.var_shape[i], dtype=np.float32) for i in range(worker.var_size)]
    ph_bucket = [tf.compat.v1.placeholder(shape=worker.var_shape[i], dtype=tf.float32) for i in range(worker.var_size)]
    bucket_assign = [tf.compat.v1.assign(worker.var_bucket[i], ph_bucket[i]) for i in range(worker.var_size)]
    
    for step in range(epoch):
        batch_num = int(worker.x_train.shape[0]/batch_size)
        for batch_cnt in range(batch_num):

            # Calculate gradients then send them to parameter server
            worker.work(batch_cnt)

            # Receive data from parameter server
            for i in range(worker.var_size):
                ind = i // worker.avg_var_size # parameter server's rank
                if i >= worker.var_size - worker.local_var_size:
                    ind = worker.num_ps-1
                comm.Recv([bucket[i], MPI.FLOAT], source=ind, tag=i-(ind*worker.avg_var_size))

            # Assign broadcasted values
            worker.sess.run(bucket_assign, feed_dict={ph_bucket[i]:bucket[i] for i in range(worker.var_size)})
            
            if batch_cnt % 10 == 0:
                print("Worker{} epoch: {} batch: {} accuracy: {}".format(rank,step,batch_cnt,worker.sess.run(worker.accuracy, feed_dict={worker.x: worker.x_test, worker.y_: worker.y_test, worker.keep_prob: 1.0})))
    
    end = time.clock()
    print("Worker{} final accuracy: {}".format(rank,worker.sess.run(worker.accuracy, feed_dict={worker.x: worker.x_test, worker.y_: worker.y_test, worker.keep_prob: 1.0})))
    print("Time: {}".format(end-start))
