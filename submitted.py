import os, h5py
from PIL import Image
import numpy as  np

###############################################################################
# TODO: here are the functions that you need to write
def todo_dataset_mean(X):
    '''
    mu = todo_dataset_mean(X)
    Compute the average of the rows in X (you may use any numpy function)
    X (NTOKSxNDIMS) = data matrix
    mu (NDIMS) = mean vector
    '''
    mu = np.zeros(X.shape[1])

    # This for loop will iterate over all columns of the array one at a time
    mu = np.mean(X, axis=0)
        
    return mu

def todo_center_datasets(train, dev, test, mu):
    '''
    ctrain, cdev, ctest = todo_center_datasets(train, dev, test, mu)
    Subtract mu from each row of each matrix, return the resulting three matrices.
    '''
    #print(train.shape, dev.shape, test.shape, mu.shape)
    X_train = train.copy()
    X_dev = dev.copy()
    X_test = test.copy()
    
    
    
    # subtract mu from each row in train matrix
    for row in range(train.shape[0]):
        for col in range(train.shape[1]):
            X_train[row][col] -= mu[col]

    for row in range(dev.shape[0]):
        for col in range(dev.shape[1]):
            X_dev[row][col] -= mu[col]
            
    for row in range(test.shape[0]):
        for col in range(test.shape[1]):
            X_test[row][col] -= mu[col]
        
        
    #print(X_train.shape, X_dev.shape, X_test.shape)
    return X_train, X_dev, X_test

def todo_find_transform(X):
    '''
    V, Lambda = todo_find_transform(X)
    X (NTOKS x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    V (NDIM x NTOKS) - The first NTOKS principal component vectors of X
    Lambda (NTOKS) - The  first NTOKS eigenvalues of the covariance or gram matrix of X

    Find and return the PCA transform for the given X matrix:
    a matrix in which each column is a principal component direction.
    You can assume that the # data is less than the # dimensions per vector,
    so you should probably use the gram-matrix method, not the covariance method.
    Standardization: Make sure that each of your returned vectors has unit norm,
    and that its first element is non-negative.
    Return: (V, Lambda)
      V[:,i] = the i'th principal component direction
      Lambda[i] = the variance explained by the i'th principal component

    V and Lambda should both be sorted so that Lambda is in descending order of absolute
    value.  Notice: np.linalg.eig doesn't always do this, and in fact, the order it generates
    is different on my laptop vs. the grader, leading to spurious errors.  Consider using 
    np.argsort and np.take_along_axis to solve this problem, or else use np.linalg.svd instead.
    '''
    U, S, V = np.linalg.svd(X, full_matrices=False)
    #print(U.shape, S.shape, V.shape)
    #X_t = np.matmul(np.matmul(U, np.diag(S)), V)
    #print(X_t.shape)
    #X_t_transposed = np.transpose(X_t)
    V_tilde = np.matmul(X.T, U)

    
    for col in range(V_tilde.shape[1]):
        col_norm = np.sqrt(sum(np.square(V_tilde[:,col])))
        for row in range(V_tilde.shape[0]):
            V_tilde[row][col] = V_tilde[row][col]/col_norm 
    
    for col in range(V_tilde.shape[1]):
        for row in range(V_tilde.shape[0]):
            if V_tilde[0][col] < 0:
                V_tilde[0][col] = -(V_tilde[0][col])
                if row < V_tilde.shape[0]-1:
                    V_tilde[row+1][col] = -(V_tilde[row+1][col])
                    
    Lambda = S**2
    
    
    return V_tilde, Lambda

def todo_transform_datasets(ctrain, cdev, ctest, V):
    '''
    ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, V)
    ctrain, cdev, ctest are each (NTOKS x NDIMS) matrices (with different numbers of tokens)
    V is an (NDIM x K) matrix, containing the first K principal component vectors
    
    Transform each x using transform, return the resulting three datasets.
    '''
    ttrain = np.matmul(ctrain, V)
    tdev = np.matmul(cdev, V)
    ttest = np.matmul(ctest, V)
    
    return ttrain, tdev, ttest

def todo_distances(train,test,size):
    '''
    D = todo_distances(train, test, size)
    train (NTRAINxNDIM) - one training vector per row
    test (NTESTxNDIM) - one test vector per row
    size (scalar) - number of dimensions to be used in calculating distance
    D (NTRAIN x NTEST) - pairwise Euclidean distances between vectors

    Return a matrix D such that D[i,j]=distance(train[i,:size],test[j,:size])
    '''
    D = np.zeros((train.shape[0], test.shape[0]))
    #print(D.shape)
    for row in range(train.shape[0]):
        for col in range(test.shape[0]):
            D[row][col] = np.linalg.norm(train[row, :size] - test[col, :size])
            
    return D

def todo_nearest_neighbor(Ytrain, D):
    '''
    hyps = todo_nearest_neighbor(Ytrain, D)
    Ytrain (NTRAIN) - a vector listing the class indices of each token in the training set
    D (NTRAIN x NTEST) - a matrix of distances from train to test vectors
    hyps (NTEST) - a vector containing a predicted class label for each test token

    Given the dataset train, and the (NTRAINxNTEST) matrix D, returns
    an int numpy array of length NTEST, specifying the person number (y) of the training token
    that is closest to each of the NTEST test tokens.
    '''
    hyps = np.zeros((D.shape[1]))

    D_trans = D.transpose()
    #print(D_trans.shape, Ytrain.shape, hyps.shape)
    index = 0
    for row in D_trans:
        Ytrain_index = np.argmin(row)
        #print(Ytrain_index, Ytrain[Ytrain_index])
        hyps[index] = Ytrain[Ytrain_index]
        index += 1
        
    return hyps

def todo_compute_accuracy(Ytest, hyps):
    '''
    ACCURACY, CONFUSION = todo_compute_accuracy(TEST, HYPS)
    TEST (NTEST) - true label indices of each test token
    HYPS (NTEST) - hypothesis label indices of each test token
    ACCURACY (scalar) - the total fraction of hyps that are correct.
    CONFUSION (4x4) - confusion[ref,hyp] is the number of class "ref" tokens (mis)labeled as "hyp"
    '''
    K = len(np.unique(Ytest)) # Number of classes 

    C_ij = np.zeros((K,K))

    for col in range(len(Ytest)):
        #print(Ytest[col], int(hyps[col]))
        C_ij[Ytest[col]][int(hyps[col])] += 1

    #print(C_ij)
    
    C_ii = np.zeros((K,K))

    for col in range(len(Ytest)):
        #print(Ytest[col], int(hyps[col]))
        C_ii[Ytest[col]][Ytest[col]] += 1

    C_ii_diag = []
    for i in range(K):
        C_ii_diag.append(C_ij[i][i])
    
    C_ii_sum = np.sum(C_ii_diag)
    
    #print(C_ii_sum)
    #print(C_ii)
    #accuracy = (Ytest == hyps).sum() / float(len(Ytest))
    accuracy = C_ii_sum / (np.sum(C_ij))
    
    acc = (round(accuracy,2))

    return acc, C_ij
    
def todo_find_bestsize(ttrain, tdev, Ytrain, Ydev, variances):
    '''
    BESTSIZE, ACCURACIES = todo_find_bestsize(TTRAIN, TDEV, YTRAIN, YDEV, VARIANCES)
    TTRAIN (NTRAINxNDIMS) - training data, one vector per row, PCA-transformed
    TDEV (NDEVxNDIMS)  - devtest data, one vector per row, PCA-transformed
    YTRAIN (NTRAIN) - true labels of each training vector
    YDEV (NDEV) - true labels of each devtest token
    VARIANCES - nonzero eigenvectors of the covariance matrix = eigenvectors of the gram matrix

    BESTSIZE (scalar) - the best size to use for the nearest-neighbor classifier
    ACCURACIES (NTRAIN) - accuracy of dev classification, as function of the size of the NN classifier

    The only sizes you need to test (the only nonzero entries in the ACCURACIES
    vector) are the ones where the PCA features explain between 92.5% and
    97.5% of the variance of the training set, as specified by the provided
    per-feature variances.  All others should be zero.
    ''' 
    list_K = []
    
    sum_variances = sum(variances)
    #print(sum_variances)
    #print(PV)
    accuracies = np.zeros((ttrain.shape[0]))
    
    for i in range(1, ttrain.shape[0]+1):
        num = 0
        for x in range(0, i):
            num += variances[x]
        PV = 100 * num / sum_variances
        if (92.5 < PV < 97.5):
            Dtest = todo_distances(ttrain, tdev, i)
            hypstest = todo_nearest_neighbor(Ytrain, Dtest)
            accuracytest,_ = todo_compute_accuracy(Ydev, hypstest)
            accuracies[i-1] = accuracytest 
            list_K.append(i)
    K = np.argmax(accuracies)

    print(accuracies)
    
    return K, accuracies

