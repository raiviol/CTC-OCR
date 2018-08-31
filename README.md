# Optical character recognition classifier with ctc-loss

This is a fully convolutional text classifier that uses [Connectionist Temporal Classification](http://www.cs.toronto.edu/%7Egraves/icml_2006.pdf).

The classifier doesn't include any recurrent or linear layers in it, but models sequences with only convolutional layers, which makes it fast and easy to train. A [ResNet](https://arxiv.org/abs/1512.03385) model is used as a base for the network. Facebook's [rosetta system](http://www.kdd.org/kdd2018/accepted-papers/view/rosetta-large-scale-system-for-text-detection-and-recognition-in-images) utilizes a similar kind of model for reading text. [Here](https://arxiv.org/abs/1709.04303) is another example used as inspiration for this network.

Baidu Research's [warp-ctc](https://github.com/baidu-research/warp-ctc) is used to calculate the ctc-loss.

[SVHN](http://ufldl.stanford.edu/housenumbers/) and [COCO-Text](https://vision.cornell.edu/se3/coco-text-2/) -datasets are included.

## Dependencies

Other versions of some dependencies might work, but are not tested.

warp-ctc:  
Installation instructions for pyTorch bindings can be found [here](https://github.com/SeanNaren/warp-ctc).
  
Python 3.6  
PyTorch 0.4  
torchvision 0.2  
  
numpy 1.14  
opencv 3.4  
sklearn 0.19  
matplotlib 2.2  
h5py 2.8  
tqdm 4.24  
six 1.11  

CTC decoder (decoder.py) is copied from Sean Naren's [DeepSpeech repository](https://github.com/SeanNaren/deepspeech.pytorch).

## Usage
Model is trained by calling:
```bash
python main.py --{param} {value}
```
Default parameters can be found in main.py.

For information on parameters call:
```bash
python main.py --help 
```

Dataset should be in [fuel dataset](https://fuel.readthedocs.io/en/latest/h5py_dataset.html) format, for other formats you might have to edit load_data -function in util.py.

Label file should be a text file with all the labels listed on their own rows, with '_' (ctc "blank" -character) first, check DATA/labels.txt for an example.

If you wish to train with your own dataset, parameters at model.py might have to be changed for optimal results.

To visualize model output with random images, call:
```bash
random_images.py {path-to-dataset} {path-to-saved-model} {path-to-labels}
```
Default values will visualize svhn dataset with a model trained with default parameters.

## Datasets
Github doesn't allow files over 100MB to be uploaded, therefore the dataset's can't be found here.
You can dowload them form [here](https://files.fm/u/yhmwsjeu), and extract them to DATA -folder.

Datasets are made with [fuel](https://fuel.readthedocs.io/en/latest/h5py_dataset.html), but can also be created with h5py, but you will need to create a function equivalent to fuel.converters.base.fill_hdf5_file.
SVHN images are 64x64 grayscale images, cropped as tight as possible from the originals.
COCO-Text images are 32x128 grayscale, cropped with given bounding boxes from the originals, rescaled to 32px height, while preserving aspect ratio. If the images are narrower than 128px, zero padding is added to the right side, otherwise they are rescaled to 128px width. (similar to facebooks [rosetta-paper](http://www.kdd.org/kdd2018/accepted-papers/view/rosetta-large-scale-system-for-text-detection-and-recognition-in-images)).
