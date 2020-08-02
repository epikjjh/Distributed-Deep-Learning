from mnist import Model

class Single(Model):
    def __init__(self, epoch, batch_size):
        super().__init__()
        self.epoch = epoch
        self.batch_size = batch_size

    # Overriden method
    def train(self):
        for step in range(self.epoch):
            x_batch, y_batch = self.data.train.next_batch(self.batch_size)
            self.sess.run(self.train_step, feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})

        # Evaluate model
        print(self.sess.run(self.accuracy, feed_dict={self.x: self.test_x, self.y_: self.test_y_, self.keep_prob: 1.0}))


if __name__ == "__main__":
    single_test = Single(500, 50)
    single_test.train()
