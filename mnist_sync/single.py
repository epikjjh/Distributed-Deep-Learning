from model import Model

class Single(Model):
    def __init__(self, epoch, batch_size):
        super().__init__()
        self.epoch = epoch
        self.batch_size = batch_size

    # Overriden method
    def train(self):
        for step in range(self.epoch):
            batch_num = int(self.x_train.shape[0]/self.batch_size)
            for batch_cnt in range(batch_num):
                x_batch = self.x_train[self.batch_size*batch_cnt:self.batch_size*(batch_cnt+1)]
                y_batch = self.y_train[self.batch_size*batch_cnt:self.batch_size*(batch_cnt+1)]
                self.sess.run(self.train_step, feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5})
                if batch_cnt % 10 == 0:
                    print("epoch: {} batch: {} accuracy: {}".format(step,batch_cnt,self.sess.run(self.accuracy, feed_dict={self.x: self.x_test, self.y_: self.y_test, self.keep_prob: 1.0})))

        # Evaluate model
        print("final accuracy: {}".format(self.sess.run(self.accuracy, feed_dict={self.x: self.x_test, self.y_: self.y_test, self.keep_prob: 1.0})))


if __name__ == "__main__":
    single_test = Single(1, 100)
    single_test.train()
