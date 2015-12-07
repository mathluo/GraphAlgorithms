from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as spa
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from misc import Parameters


_graph_params_default_values = {'affinity': 'rbf', 'n_neighbors': None, 'Laplacian_type': 'n',
'gamma': None, 'Neig' : None, 'Eig_solver': 'arpack', 
'distance_kernel_flag': None, 'num_nystrom': None, 'neighbor_type':None}


def build_affinity_matrix(raw_data,graph_params):
    """ Build affinity matrix. Wrappers using sklearn modules
    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.
    graph_params : Parameters with fields below:

        affinity : string, array-like or callable
            affinity matrix specification. If a string, this may be one 
            of 'nearest_neighbors','rbf'. 
        n_neighbors : integer
            Number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. Ignored for ``affinity='rbf'``
        Laplacian_type : 'n', normalized, 'u', unnormalized
    Return 
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        affinity matrix
    graph_degree : return the degree of the graph

    References
    ----------

    """ 

    # compute the distance matrix
    if graph_params.affinity == 'z-p': #Z-P distance, adaptive RBF kernel, currently slow!!
        k = graph_params.n_neighbors
        dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')
        tau = np.ones([raw_data.shape[0],1])
        for i, row in enumerate(dist_matrix):
            tau[i] = np.partition(row,k)[k]
        scale = np.dot(tau, tau.T)
        temp = np.exp(-dist_matrix/np.sqrt(scale))
        for i,row in enumerate(temp):
            foo = np.partition(row,row.shape[0]-k-1)[row.shape[0]-k-1]
            row[row<foo] =0
        affinity_matrix_ = np.maximum(temp, temp.T)
    else:
        if graph_params.neighbor_type != None:
            if graph_params.neighbor_type == 'connectivity':
                connectivity = kneighbors_graph(raw_data, n_neighbors=graph_params.n_neighbors, include_self=True)
                affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
                affinity_matrix_[affinity_matrix_ == 1.] = 0. 
                graph_degree = affinity_matrix_.sum(axis = 0)
                return affinity_matrix_, graph_degree                

                           
            elif graph_params.neighbor_type == 'distance':
                distance_matrix = kneighbors_graph(raw_data, n_neighbors=graph_params.n_neighbors, include_self=True, mode = 'distance')
                distance_matrix = distance_matrix*distance_matrix # square the distance
                dist_matrix = 0.5 * (distance_matrix + distance_matrix.T)
                dist_matrix = np.array(dist_matrix.todense())

        else:
            dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')

        if graph_params.affinity == 'rbf':
            gamma = graph_params.gamma
            affinity_matrix_ = np.exp(-gamma*dist_matrix)

    affinity_matrix_[affinity_matrix_ == 1.] = 0. 
    d_mean = np.mean(np.sum(affinity_matrix_,axis = 0))
    affinity_matrix_ = affinity_matrix_/d_mean
    graph_degree = affinity_matrix_.sum(axis = 0)
    return affinity_matrix_, graph_degree




def affinity_matrix_to_laplacian(W, mode = 'n'): 
    """ Build normalized Laplacian Matrix from affinity matrix W
    For dense Laplaians only. For sparse Laplacians use masks

    Parameters
    -----------
    W : affinity_matrix_

    """
    if spa.issparse(W):
        W = np.array(W.todense())  # currently not handeling sparse matrices separately, converting it to full
    if mode == 'n' : # normalized laplacian
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
    if mode == 'u' : # unnormalized laplacian
        n_nodes = W.shape[0]
        Lap = W.copy()
        # set diagonal to zero
        Lap.flat[::n_nodes + 1] = 0
        d = Lap.sum(axis=0)
        Lap = np.diag(d) - Lap     
        return Lap

def build_laplacian_matrix(raw_data,graph_params):
    """ Wrapper for building the normalized Laplacian directly from raw data

    """
    W , graph_degree = build_affinity_matrix(raw_data , graph_params)
    if graph_params.Laplacian_type == 'n':
        return affinity_matrix_to_laplacian(W,mode = 'n'), graph_degree
    else:
        return affinity_matrix_to_laplacian(W,mode = 'u'), graph_degree

def generate_eigenvectors(L,Neig):
    """ short hand for using scipy arpack package

    Parameters
    -----------
    L : laplacian_matrix_

    """
    return eigsh(L,Neig,which = 'SM')    







    


    
