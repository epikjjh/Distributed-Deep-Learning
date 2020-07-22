from typing import List
from mnist import Model
from mpi4py import MPI
import numpy as np
import tensorflow as tf
import time

class SyncWorker(Model):
    def __init__(self, comm, rank, batch_size):
        super().__init__()
        
        self.comm = comm

        # Rank 0: parameter server
        # Rank 1,2: worker 
        self.rank = rank
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


    def update(self, vars):
            apply_gradients = self.optimizer.apply_gradients((
                (vars[0], self.w_conv1), (vars[1], self.b_conv1),
                (vars[2], self.w_conv2), (vars[3], self.b_conv2),
                (vars[4], self.w_fc1), (vars[5], self.b_fc1),
                (vars[6], self.w_fc2), (vars[7], self.b_fc2),
            ))
            self.sess.run(apply_gradients)


class ParameterServer():
    def __init__(self, comm, rank):
        self.comm = comm
        super().__init__()

        # Rank 0: parameter server
        # Rank 1,2: worker 
        self.rank = rank


    def sync(self) -> List:
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

        return [
            conv1_gw, conv1_gb, conv2_gw, conv2_gb,
            fc1_gw, fc1_gb, fc2_gw, fc2_gb
        ]


if __name__ == "__main__":
    training_time = 1
    batch_size = 100
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    ps = ParameterServer(comm, rank) 
    w1 = SyncWorker(comm, rank, batch_size)
    w2 = SyncWorker(comm, rank, batch_size) 

    conv1_gw = 0.0
    conv1_gb = 0.0
    conv2_gw = 0.0
    conv2_gb = 0.0
    fc1_gw = 0.0
    fc1_gb = 0.0
    fc2_gw = 0.0
    fc2_gb = 0.0

    # Mapping variables
    vars = [
        conv1_gw, conv1_gb, conv2_gw, conv2_gb,
        fc1_gw, fc1_gb, fc2_gw, fc2_gb
    ]
        
    # Measure time
    if rank == 0:
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


