from model import Model
from mpi4py import MPI
from typing import List
import numpy as np
import tensorflow as tf
import time,sys

class SyncWorker(Model):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def work(self, cnt) -> List:
        x_batch = self.x_train[self.batch_size*cnt:self.batch_size*(cnt+1)]
        y_batch = self.y_train[self.batch_size*cnt:self.batch_size*(cnt+1)]

        # Compute gradient values
        ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        var = [var for grad, var in ret]

        # ret -> ((gw, w), (gb, b))
        grads = [grad for grad, var in ret]
        # Tuple: (gradient, variable)
        # Pack gradeint values 
        # Data: [gw_conv1, gb_conv1, gw_conv2, gb_conv2, gw_fc1, gb_fc1, gw_fc2, gb_fc2]
        # Send data to parameter server
        return grads


if __name__ == "__main__":
    epoch = 1
    batch_size = 100
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    start = time.clock()
    worker = SyncWorker(batch_size) 

    # Send parameters to parameter server
    if rank == 1:
        data = {"size": worker.var_size, "shape": worker.var_shape, "total_batch": worker.x_train.shape[0]}
        comm.send(data, dest=0, tag=0)

    # For broadcasting 
    bucket = [np.empty(worker.var_shape[i], dtype=np.float32) for i in range(worker.var_size)]
    ph_bucket = [tf.compat.v1.placeholder(shape=worker.var_shape[i], dtype=tf.float32) for i in range(worker.var_size)]

    bucket_assign = [tf.compat.v1.assign(worker.var_bucket[i], ph_bucket[i]) for i in range(worker.var_size)]

    for step in range(epoch):
        batch_num = int(worker.x_train.shape[0]/batch_size)
        for batch_cnt in range(batch_num):
            grads = worker.work(batch_cnt)
            # Send worker 1's grads
            for i in range(worker.var_size):
                comm.Send([grads[i], MPI.FLOAT], dest=0, tag=i+1)

            # Receive data from parameter server
            for i in range(worker.var_size):
                comm.Bcast([bucket[i], MPI.FLOAT], root=0) 
            # Assign broadcasted values
            worker.sess.run(bucket_assign, feed_dict={ph_bucket[i]:bucket[i] for i in range(worker.var_size)})
            if batch_cnt % 10 == 0:
                print("Worker{} epoch: {} batch: {} accuracy: {}".format(rank,step,batch_cnt,worker.sess.run(worker.accuracy, feed_dict={worker.x: worker.x_test, worker.y_: worker.y_test, worker.keep_prob: 1.0})))
    
    end = time.clock()
    print("Worker{} final accuracy: {}".format(rank,worker.sess.run(worker.accuracy, feed_dict={worker.x: worker.x_test, worker.y_: worker.y_test, worker.keep_prob: 1.0})))
    print("Time: {}".format(end-start))
