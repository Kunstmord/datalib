from os import path

import numpy as np
from skimage.io import imread
import skimage.filter
from skimage.morphology import convex_hull_image
from src.dataset import UnlabeledDataSet, LabeledDataSet


def chull(fpath):
    img = imread(fpath, as_grey=True)
    thresh = skimage.filter.threshold_otsu(img)
    binary_img = img > thresh

    chull_img = convex_hull_image(binary_img)
    return 1. * np.count_nonzero(chull_img) / (img.shape[0] * img.shape[1])


def move_to(name):
    datapath = path.join(path.dirname(path.realpath(__file__)), path.pardir)
    datapath = path.join(datapath, '../gzoo_data', 'images', name)
    print path.normpath(datapath)
    return path.normpath(datapath)


def labels():
    datapath = path.join(path.dirname(path.realpath(__file__)), path.pardir)
    datapath = path.join(datapath, '../gzoo_data', 'train_solution.csv')
    return path.normpath(datapath)

testset_path = move_to('test')
trainset_path = move_to('train')
labelspath = labels()
testdata = UnlabeledDataSet(testset_path, path.dirname(path.realpath(__file__)), file_suffix='.jpg')
traindata = LabeledDataSet(trainset_path, path.dirname(path.realpath(__file__)), labelspath, file_suffix='.jpg')
testdata.prepopulate()
traindata.prepopulate()

traindata.extract_feature(chull)
testdata.extract_feature(chull)

t_features = traindata.return_features_numpy('all')
print t_features.shape