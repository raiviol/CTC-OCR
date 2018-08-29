import os

from contextlib import redirect_stdout
from argparse import ArgumentParser

import torch
import torch.optim as optim

from historylogger import HistoryLogger
from util import load_data, load_labels
from model import Model        


def main(data_path, out_path, labelfile, resume, batch_size, epochs, resnet_depth, train):
    try:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        training_data, validation_data = load_data(data_path)
        imChannels, _, _ = training_data[0][0].shape

        # load labels in decoder format
        labels, alpha_len = load_labels(labelfile)       

        if resume is None: # start training from beginning
            model = Model(out_path, resnet_depth, imChannels, alpha_len, labels)

            # weight_decay == l2 lambda
            # SGD tends to get better end-results
            # Learning-rate is reduced on plateau, check model.py for details.
            optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.2)
            #optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.2)

            # save the  summary of the model
            with open(out_path + os.sep + "modelsummary_ctc.txt", 'w') as f:
                with redirect_stdout(f):
                    print(str(model))

        else: # resume from checkpoint
            if resume == "best":
                print("Resuming from best checkpoint")
                checkpoint = torch.load(os.path.join(out_path, 'checkpoint_best.pth.tar'))
            else:
                print("Resuming from last checkpoint")
                checkpoint = torch.load(os.path.join(out_path, 'checkpoint.pth.tar'))

            model = Model(checkpoint['model_params']['output'],
                          checkpoint['model_params']['resnet_depth'], 
                          checkpoint['model_params']['imChannels'], 
                          checkpoint['model_params']['alphabet_length'],
                          checkpoint['model_params']['labels'],
                          last_epoch = checkpoint['epoch'])
        
            model.load_state_dict(checkpoint['model_states'])

            optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.2)
            #optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.2)

            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # optimizer states have to be moved to GPU manually
            if torch.cuda.is_available():
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        # logger visualizes training process, can be followed during training
        logger = HistoryLogger(model, out_path, batch_size, epochs,
                               validation_data[0], validation_data[1],
                               training_data[0], training_data[1])

        if train:
            model.fit(training_data, validation_data, optimizer, 
                      batch_size=batch_size, epochs=epochs, logger=logger)

        torch.save(model, out_path + os.sep + "model_final.pth.tar")

    except Exception as err:
        print(err.args)
        raise 


if __name__ == '__main__':
    parser = ArgumentParser()

    data_default   = os.path.join("DATA", "svhn_ctc_extra.hdf5")
    label_default  = os.path.join("DATA", "labels.txt")
    output_default = os.path.join("results", "svhn1")
    batch_default  = 32
    epochs_default = 100
    resnet_default = 34

    parser.add_argument("--train", type=bool, default=True, dest="train", 
                        help="Bool, whether to train the model or not")

    parser.add_argument("--data", type=str, default=data_default,
                        dest="data_path", help="Path to a h5py dataset file")

    parser.add_argument("--labels", type=str, dest="labelfile",
                        default=label_default, help="Path to label mapping")

    parser.add_argument("--output", type=str, dest="out_path",
                        default=output_default, help="Output directory")

    parser.add_argument("--resume", type=str, dest="resume", default=None,
                        help="Resume from best/last checkpoint")

    parser.add_argument("--batch", type=int, dest="batch_size",
                        default=batch_default, help="Minibatch size")

    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=epochs_default, help="Number of epochs")

    parser.add_argument("--resnet", type=int, dest="resnet_depth",
                        default=resnet_default, 
                        help="ResNet version to use as base model (18, 34 or 50)")
    
    args = parser.parse_args()

    main(**vars(args))
    
