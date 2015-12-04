from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as spa
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.random import permutation

def build_affinity_matrix(raw_data,affinity,n_neighbors = None, kernel_params = {}):
    """ Build affinity matrix. Wrappers using sklearn modules
    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.
    affinity : string, array-like or callable
        affinity matrix specification. If a string, this may be one 
        of 'nearest_neighbors','rbf'. 
    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``
    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.
    Return 
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        affinity matrix
    References
    ----------

    """ 
    if affinity == 'nearest_neighbors': # k-nearest neighbor graph 
        connectivity = kneighbors_graph(raw_data, n_neighbors=n_neighbors, include_self=True)
        affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
    elif affinity == 'rbf': # full graph from rbf kernel
        if(not kernel_params):
            kernel_params['gamma'] = 1.
        affinity_matrix_ = pairwise_kernels(raw_data, metric=affinity,
                                                 filter_params=True,
                                                 **kernel_params)
    return affinity_matrix_ 

def affinity_matrix_to_laplacian(W): 
    """ Build normalized Laplacian Matrix from affinity matrix W
    For dense Laplaians only. For sparse Laplacians use masks

    Parameters
    -----------
    W : affinity_matrix_

    """
    if spa.issparse(W):
        W = np.array(W.todense())  # currently not handeling sparse matrices separately, converting it to full
    n_nodes = W.shape[0]
    Lap = -W.copy()
    # set diagonal to zero
    Lap.flat[::n_nodes + 1] = 0
    d = -Lap.sum(axis=0)
    d = np.sqrt(d)
    d_zeros = (d == 0)
    d[d_zeros] = 1
    Lap /= d
    Lap /= d[:, np.newaxis]
    Lap.flat[::n_nodes + 1] = (1 - d_zeros).astype(Lap.dtype)
    return Lap

def build_laplacian_matrix(raw_data,affinity,n_neighbors = None, kernel_params = {}):
    """ Wrapper for building the normalized Laplacian directly from raw data

    """
    W = build_affinity_matrix(raw_data , affinity,n_neighbors = n_neighbors, kernel_params = kernel_params)
    return affinity_matrix_to_laplacian(W)

def generate_eigenvectors(L,Neig):
    """ short hand for using scipy arpack package

    Parameters
    -----------
    L : laplacian_matrix_

    """
    return eigsh(L,Neig,which = 'SM')    


def generate_initial_value_binary(opt = 'rd_equal', V = None, n_samples = None):
    """  generate initial value for binary classification. 
    individual values are -1, 1 valued. 

    Parameters
    -----------
    opt: string :{'rd_equal','rd','eig'}
        options for generating values
    V: ndarray (n_samples, Neig)
        Eigenvector to generate initial condition. 
    n_samples : int
        number of nodes in the graph

    """    
    if opt == 'rd_equal':
        ind = permutation(n_samples)
        u_init = np.zeros(n_samples)
        mid = n_samples/2
        u_init[ind[:mid]] = 1
        u_init[ind[mid:]] = -1
        return u_init
    elif opt == 'rd':
        u_init = np.random.uniform(low = -1, high = 1, size = n_samples)
        return u_init
    elif opt == 'eig':
        return V[:,2].copy()





    


    
