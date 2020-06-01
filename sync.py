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
                data1 = comm.recv(source=1)
                data2 = comm.recv(source=2)

                # Apply gradients 
                gw = data1['gw'] + data2['gw']
                gb = data1['gb'] + data2['gb']
                apply_gradients = self.optimizer.apply_gradients(((gw, self.w), (gb, self.b)))
                self.sess.run(apply_gradients)

                # Broadcast gradients
                bcast_data = {'w': self.w.eval(session=self.sess), 'b': self.b.eval(session=self.sess)}
            # Worker 
            else:
                x_batch, y_batch = self.data.train.next_batch(self.batch_size)
                # Compute gradient values
                ret, = self.sess.run([self.grads], feed_dict={self.x: x_batch, self.y_: y_batch})
                
                # ret -> ((gw, w), (gb, b))
                grads = [grad for grad, var in ret]

                # Tuple: (gradient, variable)
                # Pack gradeint values 
                # Data: {'gw': gw, 'gb': gb}
                send_data = {'gw': grads[0], 'gb': grads[1]}
                
                # Send data to parameter server
                comm.send(send_data, dest=0)
                
                # Set broadcasted data 
                bcast_data = {'w': None, 'b': None}

          
            # Receive data from parameter server
            bcast_data = comm.bcast(bcast_data, root=0) 
            
            # Update variables
            if self.rank != 0:
                w = tf.assign(self.w, bcast_data['w'])
                b = tf.assign(self.b, bcast_data['b'])
                self.sess.run([w, b])

        # Evaluate model
        if self.rank == 0:
            print(self.sess.run(self.accuracy, feed_dict={self.x: self.test_x, self.y_: self.test_y_}))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    sync_test = Sync(comm, rank, 1000, 100)
    sync_test.train()
        
    
