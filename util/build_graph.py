#--------------------------------------------------------------------------
# Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
#
# This file is part of the diffuse-interface graph algorithm code. 
# There are currently no licenses. 
#
#--------------------------------------------------------------------------
# Description: 
#
#           Build Weight Matrix and Graph Laplacians        
#
#--------------------------------------------------------------------------

from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as spa
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from misc import Parameters
from nystrom import nystrom


_graph_params_default_values = {'affinity': 'rbf', 'n_neighbors': None, 
'Laplacian_type': 'n','gamma': None, 'Neig' : None, 'Eig_solver': 'full', 
'num_nystrom': None, 'neighbor_type':'full','laplacian_matrix_':None }


def build_affinity_matrix(raw_data,graph_params):
    """ Build affinity matrix. Wrappers using sklearn modules
    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.
    graph_params : Parameters with fields below:

        affinity : string
            'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
            'z-p' : adaptive kernel 
                specify n_neighbors
            '0-1' : return an unweighted graph 
                specify n_neighbors
        gamma : double
            width of the rbf kernel
        n_neighbors : integer
            Number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. 
        neighbor_type : string. 
            'full' 'knearest'
        Laplacian_type : 'n', normalized, 'u', unnormalized
    Return 
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        affinity matrix

    """ 

    # compute the distance matrix
    affinity_matrix_ = None
    if graph_params.affinity == 'z-p': #Z-P distance, adaptive RBF kernel, currently slow!!
        if graph_params.n_neighbors is None:
            raise ValueError("Please Specify number nearest points in n_neighbors")
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
        if graph_params.neighbor_type != 'full':
            if graph_params.affinity == '0-1':
                connectivity = kneighbors_graph(raw_data, n_neighbors=graph_params.n_neighbors, include_self=True)
                affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
                affinity_matrix_[affinity_matrix_ == 1.] = 0. 
                return affinity_matrix_               

                           
            elif graph_params.neighbor_type == 'knearest':
                distance_matrix = kneighbors_graph(raw_data, n_neighbors=graph_params.n_neighbors, include_self=True, mode = 'distance')
                distance_matrix = distance_matrix*distance_matrix # square the distance
                dist_matrix = 0.5 * (distance_matrix + distance_matrix.T)
                dist_matrix = np.array(dist_matrix.todense())

        else:
            dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')

        if graph_params.affinity == 'rbf':
            gamma = None
            if graph_params.gamma is None:
                print("graph kernel width gamma not specified, using default value 1")
                gamma = 1
            else : 
                gamma = graph_params.gamma
            affinity_matrix_ = np.exp(-gamma*dist_matrix)

    affinity_matrix_[affinity_matrix_ == 1.] = 0. 
    d_mean = np.mean(np.sum(affinity_matrix_,axis = 0))
    affinity_matrix_ = affinity_matrix_/d_mean
    return affinity_matrix_




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
    W  = build_affinity_matrix(raw_data , graph_params)
    if graph_params.Laplacian_type == 'n':
        return affinity_matrix_to_laplacian(W,mode = 'n')
    else:
        return affinity_matrix_to_laplacian(W,mode = 'u')

def generate_eigenvectors(L,Neig):
    """ short hand for using scipy arpack package

    Parameters
    -----------
    L : laplacian_matrix_

    """
    return eigsh(L,Neig,which = 'SM')    







class BuildGraph(Parameters):
    """ Class for graph construction and computing the graph Laplacian

    keyword arguments : 
    -----------
    Eig_solver : string
        'full' : compute the full Laplacian matrix 
        'nystrom' : specify Neig, num_nystrom(number of samples), gamma
        'arpack' : specify Neig
    (--Not using Nystrom)
        affinity : string
            'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
            'z-p' : adaptive kernel 
                specify n_neighbors
            '0-1' : return an unweighted graph 
                specify n_neighbors
        gamma : double
            width of the rbf kernel
        n_neighbors : integer
            Number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. 
        neighbor_type : string. 
            'full' 'knearest'
    (--Using Nystrom)
        affinity : only 'rbf' 
        gamma : required
        num_nystrom : number of sample points
    Laplacian_type : 'n', normalized, 'u', unnormalized

    Methods :
    -----------    
    build_Laplacian(raw_data)
    """ 
    def __init__(self, **kwargs): #interface for specifically setting graph parameters
        self.set_parameters(**kwargs)
        self.set_to_default_parameters(_graph_params_default_values)
        if self.affinity == '0-1':
            self.neighbor_type = 'knearest'


    def build_Laplacian(self,raw_data):
        """ Build graph Laplacian

        Input : 
        -----------
        raw_data : ndarray, shape (n_samples, n_features)
            Raw input data.

        """ 
        self.laplacian_matrix_ = None
        if self.Eig_solver == 'nystrom': # add code for Nystrom Extension separately   
            if self.num_nystrom is None:
                raise ValueError("Please Provide the number of sample points in num_nystrom")
            if self.gamma is None:
                print("Warning : Kernel width gamma not provided. Using Default Estimation in Nystrom")                      
            E,V= nystrom(raw_data = raw_data, num_nystrom  = self.num_nystrom, sigma = self.gamma)
            E = E[:self.Neig]
            V = V[:,:self.Neig]
            self.laplacian_matrix_ = {'V': V, 'E': E}
        else: 
            graph_params = Parameters(**self.__dict__) # for backward compatibility with older version
            Lap = build_laplacian_matrix(raw_data = raw_data,graph_params = graph_params)
            if self.Eig_solver  == 'arpack':
                E,V = generate_eigenvectors(Lap,self.Neig)
                E = E[:,np.newaxis]
                self.laplacian_matrix_ = {'V': V, 'E': E}
                return 
            elif self.Eig_solver == 'full':
                self.laplacian_matrix_ = Lap
                return
            else:
                raise NameError("Eig_Solver Needs to be either 'nystrom', 'arpack' or 'full' ")








    


    
