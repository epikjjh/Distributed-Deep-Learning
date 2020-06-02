from mnist import Model
from mpi4py import MPI
import numpy as np
import tensorflow as tf

class Sync(Model):
    def __init__(self, comm, rank, training_time, batch_size):
        super().__init__()
        
        self.comm = comm

        # Rank 0: parameter server
        # Rank 1,2: worker 
        self.rank = rank
        
        self.training_time = training_time
        self.batch_size = batch_size

    # Overriden method
    def train(self):
        for step in range(self.training_time):
            # Parameter Server
            if self.rank == 0:
                # Receive data from workers
                '''
                    Shape
                        gw: (784,10)
                        gb: (10,)
                '''
                d1_gw = np.empty((784, 10), dtype=np.float32) 
                d1_gb = np.empty(10, dtype=np.float32) 
                d2_gw = np.empty((784, 10), dtype=np.float32) 
                d2_gb = np.empty(10, dtype=np.float32) 
                # From worker1
                comm.Recv([d1_gw, MPI.DOUBLE], source=1, tag=1)
                comm.Recv([d1_gb, MPI.DOUBLE], source=1, tag=2)
                # From worker2
                comm.Recv(d2_gw, source=2, tag=1)
                comm.Recv(d2_gb, source=2, tag=2)

                # Apply gradients 
                gw = d1_gw + d2_gw
                gb = d1_gb + d2_gb
                apply_gradients = self.optimizer.apply_gradients(((gw, self.w), (gb, self.b)))
                self.sess.run(apply_gradients)

                # Broadcast gradients
                bcast_gw = self.w.eval(session=self.sess)
                bcast_gb = self.b.eval(session=self.sess)

            # Worker 
            else:
                x_batch, y_batch = self.data.train.next_batch(self.batch_size)
                # Compute gradient values
                ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch})
                
                # ret -> ((gw, w), (gb, b))
                grads = [grad for grad, var in ret]

                # Tuple: (gradient, variable)
                # Pack gradeint values 
                # Data: [gw, gb]
                
                
                # Send data to parameter server
                comm.Send([grads[0], MPI.DOUBLE], dest=0, tag=1)
                comm.Send([grads[1], MPI.DOUBLE], dest=0, tag=2)
                
                # Set broadcasted data 
                bcast_gw = np.empty((784, 10), dtype=np.float32)
                bcast_gb = np.empty(10, dtype=np.float32)

          
            # Receive data from parameter server
            comm.Bcast([bcast_gw, MPI.DOUBLE], root=0) 
            comm.Bcast([bcast_gb, MPI.DOUBLE], root=0) 
            
            # Update variables
            if self.rank != 0:
                w = tf.assign(self.w, bcast_gw)
                b = tf.assign(self.b, bcast_gb)
                self.sess.run([w, b])

        # Evaluate model
        if self.rank == 0:
            print(self.sess.run(self.accuracy, feed_dict={self.x: self.test_x, self.y_: self.test_y_}))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    sync_test = Sync(comm, rank, 1000, 100)
    sync_test.train()
        
    
