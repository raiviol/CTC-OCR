# TODO: visualize saliency?

'''
For visualizing the model output.

Press spacebar to get next image, and q to quit.
'''

import os, sys
import h5py
import cv2
import torch
import numpy as np
from numpy.random import randint


def load_data(filename):
    f = h5py.File(filename, 'r')

    features  = f['features']
    labels    = f['labels']

    splits = f.attrs['split']
    
    for s in splits:
        which_set = s[0].decode('utf-8')
        datatype  = s[1].decode('utf-8')

        if which_set == "test":
            if datatype == "features":
                x_te_idx = (s[2], s[3])
            elif datatype == "labels":
                y_te_idx = (s[2], s[3])

    x_te = features[x_te_idx[0]:x_te_idx[1],...]
    y_te = labels[y_te_idx[0]:y_te_idx[1],...]

    return x_te, y_te

def get_random_image(X, Y, label_map):
    
    r = randint(0, len(Y))

    true = ""
    for c in Y[r,...]:
        if label_map[c] != '_':
            true += label_map[c]
    
    return X[r,...,0].astype(np.uint8), true


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "DATA/svhn_ctc.hdf5"
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    else:
        model_path = "model_final.pth.tar"
    if len(sys.argv) > 3:
        label_file = sys.argv[3]
    else:
        label_file = "DATA/labels.txt"

    model = torch.load(model_path)

    label_map = {}
    with open(label_file) as fp:
        for idx, name in enumerate(fp):
            label_map[int(idx)] = name.rstrip()

    X, Y = load_data(path)
    while True:

        img, true = get_random_image(X, Y, label_map)
        
        # image needs batch and channel axis for prediction
        im2 = img.copy()
        im2 = im2[np.newaxis, np.newaxis, ...] 

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.to(device)
        p = model.predict(torch.FloatTensor(im2).to(device))

        h, w = img.shape
        img = cv2.resize(img, (w*4, h*4))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Mark predicted labels over image
        text = "Pred: {}".format(np.squeeze(p)[0][0][0]) 
        cv2.putText(img, text, (2,15), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5, [0,0,255], 2)

        # Mark true labels as comparison
        text = "True: {}".format(true)    
        cv2.putText(img, text, (2,37), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1.5, [0,0,255], 2)   

        model = torch.load(model_path)

        cv2.imshow("Random image with predictions", img)
      
        key = cv2.waitKey(0)

        if  key == ord('q'):
            break
        

