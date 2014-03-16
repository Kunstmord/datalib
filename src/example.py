"""An example (for the galaxy zoo dataset)
"""
__author__ = 'georgeoblapenko'
__license__ = "GPL"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"

from os import path
from skimage.io import imread
import skimage.filter
from src.dataset import UnlabeledDataSet, LabeledDataSet


def otsu(fpath):
    """
    Returns value of otsu threshold for an image
    """
    img = imread(fpath, as_grey=True)
    thresh = skimage.filter.threshold_otsu(img)

    return thresh


def otsu_squared(fpath, features):
    """
    Returns squared value of otsu threshold for an image
    """
    return features['otsu'] ** 2


def move_to(name):
    """
    Path to image folders
    """
    datapath = path.join(path.dirname(path.realpath(__file__)), path.pardir)
    datapath = path.join(datapath, '../gzoo_data', 'images', name)
    print path.normpath(datapath)
    return path.normpath(datapath)


def labels():
    """
    Path to labels file
    """
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

traindata.extract_feature(otsu)
traindata.extract_feature_dependent_feature(otsu_squared)
testdata.extract_feature(otsu)
testdata.extract_feature_dependent_feature(otsu_squared)

train_labels = traindata.return_labels_numpy(True)

train_features = traindata.return_features_numpy('all')
test_features = testdata.return_features_numpy('all')
print train_features.shape
print test_features.shape
print train_labels.shape

single_test_feature = testdata.return_features_numpy(['otsu_squared'])