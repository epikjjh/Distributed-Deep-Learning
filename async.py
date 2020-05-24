from mnist import Model
from mpi4py import MPI

class Async(Model):
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
        for i in range(self.training_time):
            # Parameter Server
            if self.rank == 0:

            # Worker
            else:
                x_batch, y_batch = self.data.train.next_batch(self.batch_size)
                grads = self.optimizer.compute_gradients(self.cost, [self.w, self.b])
                
        


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank() 
    async_test = Async(comm, rank)
    async_test.train()
        
    
