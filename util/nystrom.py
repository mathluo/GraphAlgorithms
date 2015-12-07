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

 
def nystrom(raw_data, n_channels, graph_params): # crude version
    """ Nystrom Extension


    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_spatial_features, n_channels)
        Raw input data.
    n_channels : number of channels for the original raw_data

    graph_deg : degree parameter for the graphs. 

    graph_parameters:

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

    if graph_params.distance_kernel_flag != None:
        kernel_flag  = graph_params.distance_kernel_flag
    else:
        kernel_flag = True
    if graph_params.gamma != None:
        sigma = graph_params.gamma
    else:
        sigma = None
    if graph_params.num_nystrom != None:
        num_nystrom = graph_params.num_nystrom
    else:
        num_nystrom = graph_params.Neig*2  

    #format data to right dimensions
    width = int((np.sqrt(raw_data.shape[1]/n_channels)-1)/2)
    if kernel_flag: # spatial kernel involved
        kernel = make_kernel(width = width, n_channels = n_channels)
        scale_sqrt = np.sqrt(kernel).reshape(1,len(kernel))
    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        num_nystrom = int(min(350, num_rows/3.))
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]


    # calculating B
    other_points = num_rows - num_nystrom
    if kernel_flag:
        distb = cdist(sample_data*scale_sqrt,other_data*scale_sqrt,'sqeuclidean')
    else: 
        distb = cdist(sample_data,other_data,'sqeuclidean')
    if sigma == None:
        sigma = np.percentile(np.percentile(distb, axis = 1, q = 5),q = 40) # a crude automatic kernel
    B = np.exp(-distb/sigma).astype(np.float32)    

    # calculating A
    if kernel_flag:
        dista = cdist(sample_data*scale_sqrt,sample_data*scale_sqrt,'sqeuclidean')
        A = np.exp(-dista/sigma).astype(np.float32)
    else: 
        dista = cdist(sample_data,sample_data,'sqeuclidean')
        A = np.exp(-dista/sigma).astype(np.float32)
        #A.flat[::A.shape[0]+1] = 0

    # normalize A and B
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    graph_deg = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./graph_deg)
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

    return L,V, graph_deg





    