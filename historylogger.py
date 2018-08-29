import os, time
from matplotlib import pyplot as plt


class HistoryLogger():
    '''
    Visualizes the training process. Can be monitored during training.

    Due to Windows's file structure, training crashes if Accuracy.pdf is open 
    when this runs, so it should be closed when evaluating.

    Linux file system allows updating the pdf while it is open.
    '''
    def __init__(self, model, path, batch_size, epoch_size, X_test, y_test, X_train, y_train):

        self.model = model
        self.path = path
        self.log = os.path.join(path, "train_log.txt")
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

        # We keep track of the losses on our own in these lists:
        self.train_losses = []
        self.test_losses  = []
        self.test_errors_seq  = []
        self.test_errors_digit  = []
        self.train_errors = []
        
        self.distances = []
        
        self.total_shown = 0
        self.start_time = time.time()
        
    def LOG(self, str):
        with open(self.log, 'a') as fp:
            fp.write(str + '\n')

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        
        # We can refer to the most recent model as "self.model"
 
        t_seq_acc, _, t_loss, _ = self.model.evaluate(self.X_train, self.y_train,
                                                      self.batch_size)

        seq_acc, dig_acc, loss, dist = self.model.evaluate(self.X_test, self.y_test,
                                                           self.batch_size)

        self.train_losses.append(t_loss)
        self.test_losses.append(loss)
        
        seq_error = 1-seq_acc
        dig_error = 1-dig_acc
        t_seq_error = 1-t_seq_acc

        self.test_errors_seq.append(seq_error)
        self.test_errors_digit.append(dig_error)
        self.train_errors.append(t_seq_error)
        self.distances.append(dist)

        # Save the losses to a text file for later use:
        with open(self.path + os.sep + "losses.txt", "a") as fp:
            fp.write("%d, %.8f, %.8f, %.8f, %.8f,\n " % (epoch, t_loss, loss, seq_error, dig_error))
        
        # Create a PDF of the accuracies and losses (and whatever you want):
        plt.close('all')
        fig, ax = plt.subplots(2,1)
        
        ax[0].set_yscale('log')
        ax[0].plot(self.test_losses, 'g.', label = "Test loss (%.3f)" % loss)
        ax[0].plot(self.train_losses, 'b.', label = "Train loss (%.3f)" % t_loss)
        
        ax[0].grid(True)
        ax[0].legend(loc = "best")
        
        ax[1].plot(self.test_errors_seq, 'r.', label = "Sequence error (%.3f)" % seq_error)
        ax[1].plot(self.train_errors, 'k.', label = "Train seq. error (%.3f)" % t_seq_error)
        ax[1].plot(self.test_errors_digit, 'b.', label = "Digit error (%.3f)" % dig_error)
        ax[1].plot(self.distances, 'g.', label = "Levenshtein (%.3f)" % dist)
        ax[1].grid(True)
        ax[1].legend(loc = "best")
        
        # Save the PDF, so we can monitor it as training goes on.
        plt.savefig(self.path + os.sep + "Accuracy.pdf")

        self.total_shown += self.batch_size * self.epoch_size
        elapsed_time = time.time() - self.start_time
        FPS = float(self.total_shown) / elapsed_time

        self.LOG("EPOCH %d: Tot = %d, FPS = %.3f, Train / test loss / mae = %.3f/%.3f/%.3f" % \
                    (epoch, self.total_shown, FPS, t_loss, loss, seq_error))

        return seq_acc, dig_acc, loss, dist

