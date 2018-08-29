import os
import shutil
import h5py
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, features, labels, normalize=True, augment=False, p_aug=0.3, 
                 trans=0.1, rot=0, shear=0, brightness=0.1, contrast=0.1, saturation=0.1):

        if normalize:
            if type(features) is np.ndarray:
                tmp = features.copy()
            else:
                tmp = features.clone().detach().cpu().numpy()

            self.mean = np.mean(tmp)
            self.std = np.std(tmp)

        if augment:
            self.p_aug = p_aug

            self.jitter = transforms.ColorJitter(brightness=brightness, 
                                                 contrast=contrast, 
                                                 saturation=saturation)

            self.affine = transforms.RandomAffine(degrees=rot, 
                                                  translate=(trans,trans), 
                                                  shear=shear)

            self.toImage = transforms.ToPILImage()
            self.toTensor = transforms.ToTensor()

        self.features = features
        self.labels = labels

        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        if self.augment:
            p = np.random.uniform(0, 1)
            if p < self.p_aug: 
                x_type = X.type()

                X = self.toImage(X/255)
                X = self.jitter(X)
                X = self.affine(X)     
                X = self.toTensor(X).type(x_type)*255

        if self.normalize:
            wasTensor = False
            if type(X) is not np.ndarray:
                tensorType = X.type()
                X = X.detach().cpu().numpy()
                wasTensor = True

            X = X.astype(np.float64)

            X -= self.mean
            X /= self.std

            if wasTensor:
                X = torch.Tensor(X).type(tensorType)

        return X, y

def load_labels(filepath):
    '''
    input: path to a .txt with all labels on their own row, 
           starting with "blank" character '_'

    output:
      str labels:          labels in decoder format , ie. "_1234567890 "
      int alphabet_length: length of labels
    '''

    labels = ""
    alphabet_length = 1

    with open(filepath, 'r') as fp:
        for row in fp:
            labels += row.rstrip()
            alphabet_length += 1

    labels += ' ' # add space as final character

    return labels, alphabet_length

def load_data(filepath):
    '''
    Load data to memory. Dataset should be in fuel format, otherwise you 
    might have to edit the function. Character locations are not needed 
    in this case, but they are included anyway, in case they are needed.

    input: path to hdf5 file

    output:
      tuple(ndarray) (x_tr, y_tr): training data
      tuple(ndarray) (x_te, y_te): validation data
    '''

    f = h5py.File(filepath, 'r')

    features  = f['features']
    labels    = f['labels']
    #locations = f['locations']

    splits = f.attrs['split']
    
    for s in splits:
        which_set = s[0].decode('utf-8')
        datatype  = s[1].decode('utf-8')

        if which_set == "train":
            if datatype == "features":
                x_tr_idx = (s[2], s[3])
            elif datatype == "labels":
                y_tr_idx = (s[2], s[3])
            #elif datatype == "locations":
            #    l_tr_idx = (s[2], s[3])

        elif which_set == "test":
            if datatype == "features":
                x_te_idx = (s[2], s[3])
            elif datatype == "labels":
                y_te_idx = (s[2], s[3])
            #elif datatype == "locations":
            #    l_te_idx = (s[2], s[3])

    x_tr = features[x_tr_idx[0]:x_tr_idx[1],...]
    y_tr = labels[y_tr_idx[0]:y_tr_idx[1],...]
    #l_tr = locations[l_tr_idx[0]:l_tr_idx[1],...]

    x_te =  features[x_te_idx[0]:x_te_idx[1],...]
    y_te =  labels[y_te_idx[0]:y_te_idx[1],...]
    #l_tr = locations[l_tr_idx[0]:l_tr_idx[1],...]

    x_tr = np.moveaxis(x_tr, -1, 1)
    x_te = np.moveaxis(x_te, -1, 1)

    return (x_tr, y_tr), (x_te, y_te)

def checkpointer(state, mostAccurate, path, filename='checkpoint.pth.tar'):
    '''
    Save best and last checkpoints of the model

    input:
      dict state: (last epoch, model state dict, optimizer state dict, 
                   best accuracy, model parameters)

      bool mostAccurate: whether to save the checkpoint as best or not
      str path:          path to seva the checkpoint
      str filename:      name with which to save the checkpoint
    
    '''

    torch.save(state, os.path.join(path, filename))
    if mostAccurate:
        shutil.copyfile(os.path.join(path, filename), 
                        os.path.join(path, filename.split('.')[0] + "_best.pth.tar"))


