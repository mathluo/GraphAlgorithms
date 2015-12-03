from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as spa
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.random import permutation

def build_affinity_matrix(raw_data,affinity,n_neighbors = None, kernel_params = None):
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
        if(kernel_params == None):
            kernel_params = {}
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

def build_laplacian_matrix(raw_data,affinity,n_neighbors = None, kernel_params = None):
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

def generate_random_fidelity(ind, perc):
    """  generate perc percent random fidelity out of index set ind

    Parameters
    -----------
    ind : ndarray, (n_sample_in_class, )
    perc : float, percent to sample

    """
    ind = np.array(ind)
    num_sample = int(np.ceil(len(ind)*perc))
    ind2 = np.random.permutation(len(ind))
    return ind[ind2[:num_sample]]


def to_standard_labels(labels):
    """  convert any numeric labeling, i.e., labels that are numbers, to
    standard form, 0,1,2,...

    Parameters
    -----------
    labels : ndarray, (n_labels, )

    Return 
    -----------
    out_labels : ndarray, (n_labels, )
    """    
    tags = np.unique(labels)
    out_labels = np.zeros(labels.shape)
    for i, tag in enumerate(tags):
        out_labels[labels == tag] = i
    return out_labels

def vector_to_labels(V):
    """  convert a multiclass assignment vector (n_samples, n_class) 
    to a standard label 0,1,2... (n_samples,) by projecting onto largest component

    Parameters
    -----------
    V : ndarray, shape(n_samples,n_class)
        class assignment vector for multiclass

    """
    return np.argmax(V, axis = 1)


def labels_to_vector(labels):
    """  convert a standard label 0,1,2... (n_samples,)
    to a multiclass assignment vector (n_samples, n_class) by assigning to e_k

    Parameters
    -----------
    labels : ndarray, shape(n_samples,)

    """
    labels = labels.astype(int)
    n_class = max(labels)+1
    vec = np.zeros([labels.shape[0], n_class])
    for i in range(n_class):
        vec[labels == i,i] = 1.
    return vec

def standard_to_binary_labels(labels):
    out_labels = np.zeros(labels.shape)
    out_labels[labels == 0] = -1
    out_labels[labels == 1] = 1 
    return out_labels


def generate_initial_value_multiclass(opt , n_samples = None, n_class = None): 
    """  generate initial value for multiclass classification. 
    an assignment matrix is returned 

    Parameters
    -----------
    opt: string :{'rd_equal','rd'}
        options for generating values
    n_samples : int
        number of nodes in the graph

    Return
    -------
    u_init : ndarray, shape(n_samples, n_class)

    """   
    
    if opt == 'rd_equal':
        ind = permutation(n_samples)
        u_init = np.zeros([n_samples, n_class]) 
        sample_per_class = n_samples/n_class
        for i in range(n_class):
            u_init[ind[i*sample_per_class:(i+1)*sample_per_class], i] = 1.
        return u_init
    elif opt == 'rd':
        u_init = np.random.uniform(low = -1, high = 1, size = [n_samples,n_class])
        u_init = labels_to_vector(vector_to_labels(u_init))
        return u_init

def compute_error_rate(labels, ground_truth):
    """ compute the error rate of a classification given ground_truth and the labels 
    since for clustering the order of the labels are relative, we will automatically 
    match ground_truth with labels according to the highest percentage of population 
    in ground truth 

    Parameters
    -----------
    labels : ndarray, shape(n_samples, )
    ground_truth : ndarray, shape(n_samples, )
    """
    # format the labels
    labels = labels.astype(int)
    format_labels = np.zeros(labels.shape).astype(int)
    n_class = max(labels)
    for tag in range(n_class+1):
       format_labels[labels == tag] = np.argmax(np.bincount(ground_truth[labels == tag].astype(int)))
    return float(len(ground_truth[format_labels!= ground_truth]))/float(len(ground_truth))




    


    
