from typing import List
from mnist import Model
from mpi4py import MPI
import numpy as np
import tensorflow as tf
import time

class SyncWorker(Model):
    def __init__(self, comm, rank, training_time, batch_size):
        super().__init__()
        
        self.comm = comm

        # Rank 0: parameter server
        # Rank 1,2: worker 
        self.rank = rank
        
        self.training_time = training_time
        self.batch_size = batch_size


    def worker(self) -> List:
        x_batch, y_batch = self.data.train.next_batch(self.batch_size)
        # Compute gradient values
        ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
        
        # ret -> ((gw, w), (gb, b))
        grads = [grad for grad, var in ret]

        # Tuple: (gradient, variable)
        # Pack gradeint values 
        # Data: [gw_conv1, gb_conv1, gw_conv2, gb_conv2, gw_fc1, gb_fc1, gw_fc2, gb_fc2]
        # Send data to parameter server
        comm.Send([grads[0], MPI.DOUBLE], dest=0, tag=1)
        comm.Send([grads[1], MPI.DOUBLE], dest=0, tag=2)
        comm.Send([grads[2], MPI.DOUBLE], dest=0, tag=3)
        comm.Send([grads[3], MPI.DOUBLE], dest=0, tag=4)
        comm.Send([grads[4], MPI.DOUBLE], dest=0, tag=5)
        comm.Send([grads[5], MPI.DOUBLE], dest=0, tag=6)
        comm.Send([grads[6], MPI.DOUBLE], dest=0, tag=7)
        comm.Send([grads[7], MPI.DOUBLE], dest=0, tag=8)
        return list()



class ParameterServer():
    def __init__(self, comm, rank, training_time, batch_size):
        self.comm = comm

        # Rank 0: parameter server
        # Rank 1,2: worker 
        self.rank = rank
        
        self.training_time = training_time
        self.batch_size = batch_size

    def sync(self):
        # Receive data from workers
        # From worker1
        comm.Recv([self.d1_conv1_gw, MPI.DOUBLE], source=1, tag=1)
        comm.Recv([self.d1_conv1_gb, MPI.DOUBLE], source=1, tag=2)
        comm.Recv([self.d1_conv2_gw, MPI.DOUBLE], source=1, tag=3)
        comm.Recv([self.d1_conv2_gb, MPI.DOUBLE], source=1, tag=4)
        comm.Recv([self.d1_fc1_gw, MPI.DOUBLE], source=1, tag=5)
        comm.Recv([self.d1_fc1_gb, MPI.DOUBLE], source=1, tag=6)
        comm.Recv([self.d1_fc2_gw, MPI.DOUBLE], source=1, tag=7)
        comm.Recv([self.d1_fc2_gb, MPI.DOUBLE], source=1, tag=8)

        # From worker2
        comm.Recv([self.d2_conv1_gw, MPI.DOUBLE], source=2, tag=1)
        comm.Recv([self.d2_conv1_gb, MPI.DOUBLE], source=2, tag=2)
        comm.Recv([self.d2_conv2_gw, MPI.DOUBLE], source=2, tag=3)
        comm.Recv([self.d2_conv2_gb, MPI.DOUBLE], source=2, tag=4)
        comm.Recv([self.d2_fc1_gw, MPI.DOUBLE], source=2, tag=5)
        comm.Recv([self.d2_fc1_gb, MPI.DOUBLE], source=2, tag=6)
        comm.Recv([self.d2_fc2_gw, MPI.DOUBLE], source=2, tag=7)
        comm.Recv([self.d2_fc2_gb, MPI.DOUBLE], source=2, tag=8)

        # Apply gradients 
        conv1_gw = self.d1_conv1_gw + self.d2_conv1_gw
        conv1_gb = self.d1_conv1_gb + self.d2_conv1_gb
        conv2_gw = self.d1_conv2_gw + self.d2_conv2_gw
        conv2_gb = self.d1_conv2_gb + self.d2_conv2_gb
        fc1_gw = self.d1_fc1_gw + self.d2_fc1_gw
        fc1_gb = self.d1_fc1_gb + self.d2_fc1_gb
        fc2_gw = self.d1_fc2_gw + self.d2_fc2_gw
        fc2_gb = self.d1_fc2_gb + self.d2_fc2_gb
        apply_gradients = self.optimizer.apply_gradients((
            (conv1_gw, self.w_conv1), (conv1_gb, self.b_conv1),
            (conv2_gw, self.w_conv2), (conv2_gb, self.b_conv2),
            (fc1_gw, self.w_fc1), (fc1_gb, self.b_fc1),
            (fc2_gw, self.w_fc2), (fc2_gb, self.b_fc2),
        ))
        self.sess.run(apply_gradients)

        # Broadcast variables
        self.bcast_conv1_w = self.w_conv1.eval(session=self.sess)
        self.bcast_conv1_b = self.b_conv1.eval(session=self.sess)
        self.bcast_conv2_w = self.w_conv2.eval(session=self.sess)
        self.bcast_conv2_b = self.b_conv2.eval(session=self.sess)
        self.bcast_fc1_w = self.w_fc1.eval(session=self.sess)
        self.bcast_fc1_b = self.b_fc1.eval(session=self.sess)
        self.bcast_fc2_w = self.w_fc2.eval(session=self.sess)
        self.bcast_fc2_b = self.b_fc2.eval(session=self.sess)
        

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    sync_test = Sync(comm, rank, 2000, 100)
    sync_test.train()
        
    # Measure time
    if self.rank == 0:
        start = time.clock()

    for step in range(self.training_time):
        # Parameter Server
        if self.rank == 0:
            self.parameter_server()

        # Worker 
        else:
            self.worker()
      
        # Receive data from parameter server
        comm.Bcast([self.bcast_conv1_w, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_conv1_b, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_conv2_w, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_conv2_b, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_fc1_w, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_fc1_b, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_fc2_w, MPI.DOUBLE], root=0) 
        comm.Bcast([self.bcast_fc2_b, MPI.DOUBLE], root=0) 
        
        # Update variables (worker)
        if self.rank != 0:
            w_conv1 = tf.assign(self.w_conv1, self.bcast_conv1_w)
            b_conv1 = tf.assign(self.b_conv1, self.bcast_conv1_b)
            w_conv2 = tf.assign(self.w_conv2, self.bcast_conv2_w)
            b_conv2 = tf.assign(self.b_conv2, self.bcast_conv2_b)
            w_fc1 = tf.assign(self.w_fc1, self.bcast_fc1_w)
            b_fc1 = tf.assign(self.b_fc1, self.bcast_fc1_b)
            w_fc2 = tf.assign(self.w_fc2, self.bcast_fc2_w)
            b_fc2 = tf.assign(self.b_fc2, self.bcast_fc2_b)
            self.sess.run([
                w_conv1, b_conv1,
                w_conv2, b_conv2,
                w_fc1, b_fc1,
                w_fc2, b_fc2
            ])

    # Evaluate model
    if self.rank == 0:
        end = time.clock()
        print(self.sess.run(self.accuracy, feed_dict={self.x: self.test_x, self.y_: self.test_y_, self.keep_prob: 1.0}))
        print(end-start)


