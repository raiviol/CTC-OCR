from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau as lr_reducer
import torch.nn.functional as F
from torch.utils import data

from warpctc_pytorch import CTCLoss 
from decoder import GreedyDecoder

from sklearn.metrics import accuracy_score

from util import checkpointer, Dataset


def levenshtein(seq1, seq2):
    '''
    Levenshtein edit distance

    input:  lists/strings from which to calculate the distance
    output: distance as an int
    '''

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    M = np.zeros ((size_x, size_y))

    for x in range(size_x):
        M[x, 0] = x
    for y in range(size_y):
        M[0, y] = y

    for x in range(1,size_x):
        for y in range(1,size_y):
            if seq1[x-1] == seq2[y-1]:
                M[x,y] = min(
                    M[x-1, y] + 1,
                    M[x-1, y-1],
                    M[x, y-1] + 1
                )
            else:
                M[x,y] = min(
                    M[x-1,y] + 1,
                    M[x-1,y-1] + 1,
                    M[x,y-1] + 1
                )

    return (M[size_x - 1, size_y - 1])

class Model(nn.Module):
    def __init__(self, 
                 output,
                 resnet_depth    = 34,
                 imChannels      = 1,
                 alphabet_length = 11,
                 labels          = "_1234567890 ",
                 last_epoch      = 0
                 ):

        super(Model, self).__init__()

        self.out_path = output
        self.last_epoch = last_epoch

        # save model parameters for checkpointer
        self.modelparams = {"output"          : output,
                            "resnet_depth"    : resnet_depth,
                            "imChannels"      : imChannels,
                            "alphabet_length" : alphabet_length,
                            "labels"          : labels}

        # ----- MODEL ------

        # same as resnet input layer, but with support for grayscale input
        self.input = nn.Conv2d(imChannels, 64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(alphabet_length)

        # leave first maxpool out to get more ouput dimensions, since
        # output string maximum length is limited by this
        self.relu = nn.ReLU(inplace=True)
        
        if resnet_depth == 18:
            from torchvision.models import resnet18 as resnet
            resnet_output_size = 256

        elif resnet_depth == 34:
            from torchvision.models import resnet34 as resnet
            resnet_output_size = 256

        elif resnet_depth == 50:
            from torchvision.models import resnet50 as resnet
            resnet_output_size = 1024

        else: 
            raise NotImplementedError("Only resnet depths of 18, 34, and 50 are supported.")

        # some layers are also removed from the end, since small images 
        # result in negative filter sizes on those layers, and training crashes
        resnet_orig = resnet()
        resnet_no_top = nn.Sequential(*list(resnet_orig.children())[4:-3])

        self.resnet = resnet_no_top

        self.dropout = nn.Dropout2d()

        conv1 = nn.Conv2d(resnet_output_size, 
                          int(resnet_output_size/2), 2)
        conv2 = nn.Conv2d(int(resnet_output_size/2), 
                          int(resnet_output_size/4), 2)
        conv3 = nn.Conv2d(int(resnet_output_size/4), 
                          alphabet_length, 1)        

        # svhn and coco-text datasets are supported, you might have to change 
        # these layers if using other datasests
        if alphabet_length == 11: #svhn parameters
            self.conv_out = nn.Sequential(conv1, self.relu, self.dropout,
                                          conv2, self.relu, self.dropout, 
                                          conv3, self.bn2, self.relu)
        else: # coco-text parameters
            conv3 = nn.Conv2d(int(resnet_output_size), alphabet_length, 1) 
            self.conv_out = nn.Sequential(self.dropout, conv3, self.bn2, self.relu) 
    
        n_param = self.count_parameters()
        print("\nModel initialized, {} trainable parameters".format(n_param))

        # ------ UTILITIES ------

        self.labels = labels
        self.decoder = GreedyDecoder(self.labels)

        self.loss_fcn = CTCLoss()

        # ------ INIT DEVICES -------

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("GPU not found, using CPU ...")
        else:
            if torch.cuda.device_count() > 1:
                print("Using {} cuda devices ...".format(torch.cuda.device_count()))

                self.input     = nn.DataParallel(self.conv_out)
                self.resnet    = nn.DataParallel(self.resnet)    
                self.relu      = nn.DataParallel(self.relu)
                self.dropout   = nn.DataParallel(self.dropout)
                self.conv_out  = nn.DataParallel(self.conv_out)

            else:
                print("using {} ...".format(self.device))
        

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.input(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.resnet(x)

        x = self.conv_out(x)
        x = self.decode_conv(x)

        return x

    def decode_conv(self, x):
        '''
        Reshape model output. x is of shape Batch x Channel x Height x Width,
        which we reshape to Sequence x Batch x Channel required by warp-ctc

        Sequence axis is concatenated from height and width axes (mean over H axis).
        '''

        x = torch.mean(x, -2)
        x = x.permute(2, 0, 1)

        return x

    def remove_padding(self, labels, max_len):
        '''
        Padding isn't needed with ctc loss, but dataset currently has padding
        to make every sequence the same length.

        Also returns string lengths as a list, for warp-ctc.
        '''

        target_sizes = []
        nonzero = []
        for i in range(labels.shape[0]):
            length = 0
            # output can't be longer than max model output sequence length
            # this is also the length limit of the strings predicted
            for j in range(min(max_len, labels.shape[1])): 
                if labels[i,j] != 0:
                    length += 1
                    nonzero.append(labels[i,j])
                else:
                    break
                        
            target_sizes.append(length)

        output = torch.IntTensor(nonzero)
        output.requires_grad_(False)

        target_sizes = torch.IntTensor(target_sizes)
        target_sizes = torch.autograd.Variable(target_sizes, requires_grad=False)

        return output, target_sizes

    def fit(self, training_data, validation_data, optimizer, 
            epochs=100, batch_size=10, logger=None):
        '''
        Training loop

        input:
          tuple training_data:   (features, labels), as numpy.ndarray
          tuple validation_data: (features, labels), as numpy.ndarray
          optimizer: a PyTorch optimizer object
          logger:    a HistoryLogger object, for visualizing training
        '''

        # Learning-rate is reduced on plateau, parameters might need tweaking depending on data
        scheduler = lr_reducer(optimizer, factor=0.5, patience=5, min_lr=1e-6)

        logger.on_train_begin()

        self.to(self.device)
        
        # Data to tensors
        x_tr = torch.tensor(training_data[0]).type(torch.FloatTensor)
        y_tr = torch.IntTensor(training_data[1])
        x_te = torch.tensor(validation_data[0]).type(torch.FloatTensor)
        y_te = torch.IntTensor(validation_data[1])

        print("\nTraining with {} examples, validating with {} examples"\
              .format(x_tr.shape[0], x_te.shape[0]))

        best_acc = 0 # for determining best checkpoint

        training_set = Dataset(x_tr, y_tr, normalize=True, augment=True)

        params = {'batch_size':batch_size, 'shuffle':True, 'num_workers':6}
        train_generator = data.DataLoader(training_set, **params)

        for e in range(self.last_epoch, epochs):

            print("\nStarting epoch {}/{} ...".format(e+1, epochs))

            loss = 0

            # Progress bar
            with tqdm(train_generator,
                      total = (x_tr.shape[0]//batch_size)+1, 
                      unit_scale = batch_size,
                      postfix = "loss: {}".format(loss)) as t:

                for x_batch, y_batch in t:

                    # one batch at a time to GPU, uses less GPU memory
                    x_batch = x_batch.to(self.device)

                    optimizer.zero_grad()

                    y = self(x_batch) # predict on batch

                    # Output strings are all the same length, but warp-ctc needs 
                    # these lengths as a tensor to work.
                    out_sizes = torch.IntTensor([y.shape[0]]*batch_size)
                    out_sizes = torch.autograd.Variable(out_sizes, requires_grad=False)
                    
                    y_batch, target_sizes = self.remove_padding(y_batch, y.shape[0])

                    # predictions are needed without softmax in the loss function
                    loss = self.loss_fcn(y, y_batch, out_sizes, target_sizes)

                    loss.backward() 
                    optimizer.step()

                    t.postfix = "loss: {:8.4f}".format(loss.cpu().detach().numpy()[0])

                    torch.cuda.empty_cache()

            if logger is not None:
                seq_acc, _, loss, _ = logger.on_epoch_end(e)
            else:
                seq_acc, _, loss, _ = self.evaluate(x_te, y_te, batch_size)    

            scheduler.step(loss)

            # Save checkpoint
            most_accurate = False
            if seq_acc > best_acc:
                best_acc = seq_acc 
                most_accurate = True

            checkpointer({'epoch'        : e+1,
                          'model_states' : self.state_dict(),
                          'optimizer'    : optimizer.state_dict(),
                          'accuracy'     : best_acc,
                          'model_params' : self.modelparams}, 
                          most_accurate, self.out_path)

            print("validation accuracy: {:4.2f}, validation loss: {:6.4f}"\
                  .format(float(seq_acc), float(loss)))

    def predict(self, x):
        '''
        input:
          FloatTensor/ndarray x: input to predict ouput with

        output:
          list y:        list of strings
          FloatTensor p: raw model output, needed for ctc-loss
          out_sizes:     list of output string lengths, needed for ctc-loss
        '''
        with torch.no_grad():

            p = self(x) # predict on batch     

            out_sizes = torch.IntTensor([p.shape[0]]*p.shape[1])
            out_sizes = torch.autograd.Variable(out_sizes, requires_grad=False)

            # Reformat y to be compatible with decoder
            y = p.permute(1, 0, 2)
            y = F.softmax(y, dim=-1)

            sizes = torch.IntTensor([y.size()[1]]*y.size()[0])

            y, _ = self.decoder.decode(y, sizes)     

            torch.cuda.empty_cache()    

            return y, p, out_sizes

    def evaluate(self, x, y, batch_size):
        '''
        input:
          FloatTensor/ndarray x: data
          list[str] y: targets
          int batch_size: batch size, x and y should be dividable with this 

        output:
          seq_acc: mean accuracy, per sequence
          dig_acc: mean accuracy, per digit
          loss:    mean ctc loss
          dist:    mean Levenshtein edit distance
        '''

        print ("evaluating...", end="\r")

        with torch.no_grad():
            
            loss = 0
            dist = 0
            seq_acc = 0
            dig_acc = 0

            if type(x) is not torch.FloatTensor:
                x = torch.FloatTensor(x)

            eval_set = Dataset(x, y, normalize=True, augment=False)

            params = {'batch_size':batch_size, 'shuffle':False, 'num_workers':6}
            eval_generator = data.DataLoader(eval_set, **params)
            
            for x_batch, y_batch in eval_generator:

                x_batch = x_batch.to(self.device)

                y, p, out_sizes = self.predict(x_batch)

                y_batch_no_pad, target_sizes = self.remove_padding(y_batch, p.shape[0])

                # ------ LOSS CALCULATION ------          

                loss += self.loss_fcn(p, y_batch_no_pad, out_sizes, target_sizes).numpy()   

                # ------ ACCURACY CALCULATION ------

                y_batch = self.decoder.convert_to_strings(y_batch)

                for i in range(len(y_batch)):
                    chars_pred = list(y[i][0])
                    chars_true = list(y_batch[i][0])
                    
                    len_diff = len(chars_pred) - len(chars_true)
                    if len_diff < 0:
                        chars_pred += ["_"]*(-len_diff)
                    elif len_diff > 0:
                        chars_true += ["_"]*len_diff

                    dist    += levenshtein(chars_pred , chars_true)
                    seq_acc += accuracy_score(y[i], y_batch[i])
                    dig_acc += accuracy_score(chars_pred, chars_true)

                torch.cuda.empty_cache()

            # average over sequences
            loss    /= x.shape[0]
            dist    /= x.shape[0]
            seq_acc /= x.shape[0]
            dig_acc /= x.shape[0]

            return seq_acc, dig_acc, loss, dist


