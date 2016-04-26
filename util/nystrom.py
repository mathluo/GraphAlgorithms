#--------------------------------------------------------------------------
# Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
#
# This file is part of the diffuse-interface graph algorithm code. 
# There are currently no licenses. 
#
#--------------------------------------------------------------------------
# Description: 
#
#           Implementation of Nystrom Extension.          
#
#--------------------------------------------------------------------------



import scipy.sparse as spa
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from scipy.spatial.distance import cdist
from scipy.linalg import pinv
from scipy.linalg import sqrtm  
from scipy.linalg import eigh
from scipy.linalg import eig


def make_kernel(width = None, n_channels = None):
    """ Construct a Gaussian-like patch and flatten 
    the patch to desired format
    """
    kernel = np.zeros([2*width+1,2*width+1])
    for d in range(1,width+1):
        value= 1. / ((2*d+1.)*(2*d+1.))
        for i in range(-d, d+1):
            for j in range(-d, d+1): 
                kernel[width -i,width - j ] = kernel[ width - i,width - j ] + value
    kernel = kernel/width
    kernel = kernel.flatten()
    kernel = np.array([kernel, ]*n_channels)
    kernel = kernel.flatten()
    return kernel


def flatten_23(v): # short hand for the swapping axis
    return v.reshape(v.shape[0],-1, order = 'F')

 
def nystrom(raw_data, num_nystrom  = 300, sigma = None): # basic implementation
    """ Nystrom Extension


    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.

    sigma : width of the rbf kernel

    num_nystrom : int, 
            number of sample points 

    Return 
    ----------
    V : eigenvectors
    E : eigenvalues    
    """

    #format data to right dimensions
    # width = int((np.sqrt(raw_data.shape[1]/n_channels)-1)/2)
    # if kernel_flag: # spatial kernel involved   # depreciated. Put spatial mask in the image patch extraction process
    #     kernel = make_kernel(width = width, n_channels = n_channels)
    #     scale_sqrt = np.sqrt(kernel).reshape(1,len(kernel))
    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]


    # calculating B
    other_points = num_rows - num_nystrom
    distb = cdist(sample_data,other_data,'sqeuclidean')
    if sigma == None:
        sigma = np.percentile(np.percentile(distb, axis = 1, q = 5),q = 40) # a crude automatic kernel
    B = np.exp(-distb/sigma).astype(np.float32)    

    # calculating A
    dista = cdist(sample_data,sample_data,'sqeuclidean')
    A = np.exp(-dista/sigma).astype(np.float32)
        #A.flat[::A.shape[0]+1] = 0

    # normalize A and B
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    B = B*B1

    # do orthogonalization and eigen-decomposition
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A+ np.dot(np.dot(Asi,BBT),Asi)
    R = (R+R.transpose())/2.
    E, U = eigh(R)
    E = np.real(E)
    ind = np.argsort(E)[::-1]
    U = U[:,ind]
    E = E[ind]
    W = np.dot(W,Asi)
    V = np.dot(W, U)
    V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    V = np.real(V)
    E = 1-E

    return E,V





    