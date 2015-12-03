""" script for Nystrom Extension
"""

''' 
patch is 2*width+1 wide. 
indexing is first left to right, then up to down. 

'''
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



def imageblocks(im, width, format = 'flat'):
    """Extract all blocks of specified size from an image or list of images
    Automatically pads image to ensure num_blocks = num_pixels
    """

    # See
    # http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    if len(im.shape) == 2:
        im = im[:,:,np.newaxis]
    im = np.pad(im,[[width,width],[width,width],[0,0]] ,mode = 'symmetric' )
    Nr, Nc, n_channels = im.shape
    blksz = 2*width+1
    shape = (Nr-2*width, Nc-2*width, blksz, blksz, n_channels)
    strides = im.itemsize*np.array([Nc*n_channels, n_channels, Nc*n_channels, n_channels, 1])
    sb = np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)
    sb = np.ascontiguousarray(sb)
    sb = sb[:(Nr-2*width),:Nc-2*width,:,:,:]
    sb = np.ascontiguousarray(sb)
    sb.shape = (-1, blksz, blksz, n_channels)
    n_samples = sb.shape[0]
    if format == 'flat':
        return sb.transpose([0,3,1,2]).reshape(n_samples,n_channels,blksz*blksz).transpose([0,2,1])
    else:
        return sb



 
def nystrom(raw_data, num_nystrom , n_channels, sigma = 22.,  kernel_flag = True): # crude version
    """ Nystrom Extension


    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_spatial_features, n_channels)
        Raw input data.
    affinity : string, array-like or callable
        currently set to 'rbf'
    num_nystrom : int, 
        number of sample points 
    kernel_flag : bool
        whether to use spatial kernel 

    Return 
    ----------
    V : eigenvectors, 
    E : eigenvalues    
    """

    #format data to right dimensions
    width = int((np.sqrt(raw_data.shape[1])-1)/2)
    if kernel_flag: # spatial kernel involved
        kernel = make_kernel(width = width, n_channels = n_channels)
        scale_sqrt = np.sqrt(kernel).reshape(1,len(kernel))
    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]
    

    # calculating A
    sigma = sigma*sigma 
    if kernel_flag:
        A = np.exp(-cdist(sample_data*scale_sqrt,sample_data*scale_sqrt,'sqeuclidean')/sigma).astype(np.float32)
    else: 
        A = np.exp(-cdist(sample_data,sample_data,'sqeuclidean')/sigma).astype(np.float32)
        #A.flat[::A.shape[0]+1] = 0

    # calculating B
    other_points = num_rows - num_nystrom
    if kernel_flag:
        B = np.exp(-cdist(sample_data*scale_sqrt,other_data*scale_sqrt,'sqeuclidean')/sigma).astype(np.float32)
    else: 
        B = np.exp(-cdist(sample_data,other_data,'sqeuclidean')/sigma).astype(np.float32)

    # normalize A and B
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    dhat = np.sqrt(1./np.concatenate((d1,d2),axis = 0))
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
    L, U = eigh(R)
    L = np.real(L)
    ind = np.argsort(L)[::-1]
    U = U[:,ind]
    L = L[ind]
    W = np.dot(W,Asi)
    V = np.dot(W, U)
    V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    V = np.real(V)
    L = 1-L

    return L,V





    