import os, h5py
from PIL import Image
import numpy as  np

def classify(Xtrain, Ytrain, Xdev, Ydev, Xtest):
    '''
    Ytest = classify(Xtrain, Ytrain, Xdev, Ydev, Xtest)

    Use any technique you like to train a classifier with the training set,
    and then return the correct class labels for the test set.
    Extra credit points are provided for beating various thresholds above 50%.

    Xtrain (NTRAIN x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    Ytrain (NTRAIN) - list of class indices
    Xdev (NDEV x NDIM) - data matrix.
    Ydev (NDEV) - list of class indices
    Xtest (NTEST x NDIM) - data matrix.
    '''
    ntrain, ndim = Xtrain.shape
    ndev = len(Ydev)
    ntest = Xtest.shape[0]
    
    print(ntrain, ndim, ndev, ntest)
