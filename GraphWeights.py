'script for computing graph information'
import scipy.linalg as la
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import numpy.linalg as la
from networkx.readwrite import json_graph
import json
from scipy.spatial.distance import cdist
import scipy.sparse.linalg as spla
import scipy.sparse as sp


def norm_squared(vector):
    """Returns the L2 norm squared of a vector
    """
    
    return np.float(np.sum(np.power(vector,2)))

def inv_sqrt_degree_matrix(affinity_matrix):
    """Returns a sparse matrix D^(-1/2) where D is the degree matrix associated to the affinity_matrix
    
    Notes
    -----
    Sparsity help perform matrix operations more quickly.  We use Compressed Row Format (CSR).
    """
    
    sum_list = 1./np.sum(affinity_matrix, dtype = np.float, axis = 0)
    return sp.diags(np.sqrt(sum_list), 0, format = 'csr')


def symmetric_laplacian_gen(affinity_matrix):
    """Returns the sparse symmetric laplacian of an affinity matrix (affinity_matrix)

    Notes
    -----
    Sparsity help perform matrix operations more quickly.  We use Compressed Row Format (CSR).  See Luxemburg's Tutorial on 
    Spectral Clustering for details.  She calls this matrix L_sym.
    """
    D = inv_sqrt_degree_matrix(affinity_matrix)
    A = sp.csr_matrix(affinity_matrix)
    return sp.eye(np.shape(affinity_matrix)[0], format = 'csr') - D*A*D

def affinity_matrix_gen(X,mth_closest_point = 50):
    """Returns affinity matrix given a raw data set in Euclidean space 
    """
    #Construct a distance matrix
    distance_matrix = cdist(X,X)
    #Find the mth nearest neighbors using mth_closest_point
    mth_closest_points_list = [] #indexed by points
    index_of_distance_sorted_by_rows = np.argsort(distance_matrix)
    n = X.shape[0]
    for i in range(n):
        dist = distance_matrix[i, index_of_distance_sorted_by_rows[i, mth_closest_point]]
        mth_closest_points_list.append(dist)
        
    #Construct the affinity matrix
    affinity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in index_of_distance_sorted_by_rows[i, :mth_closest_point]: #we are excluding i = j and allowing those to be 0.
            similarity_value = float(np.exp(-float(distance_matrix[i,j])**2/np.sqrt(float(mth_closest_points_list[i])*float(mth_closest_points_list[j]))))
            affinity_matrix[i,j] = similarity_value
    affinity_matrix = np.maximum(affinity_matrix, affinity_matrix.T)
    return affinity_matrix

def smallest_eigenvv(laplacian_matrix,Neigs):
    eigenvv = spla.eigsh(laplacian_matrix, k = Neigs, which= 'SM') # note that we need the smallest eigenvalues!
    return eigenvv

class GraphWeights: 
    """ Basic functionalties for computing and constructing graphs 
    Can be used as base class for data manipulation classes
    """
    def __init__(self,X):
        self.X = X

    def symmetric_laplacian_gen(self):
        affinity_matrix = self.affinity_matrix_gen()
        return symmetric_laplacian_gen(affinity_matrix)

    def affinity_matrix_gen(self):
        return affinity_matrix_gen(self.X)  
        
    def smallest_eigenvv(self,Neigs):
        laplacian_matrix = self.symmetric_laplacian_gen()
        return smallest_eigenvv(laplacian_matrix,Neigs)
































