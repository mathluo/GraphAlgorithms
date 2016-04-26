#--------------------------------------------------------------------------
# Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
#
# This file is part of the diffuse-interface graph algorithm code. 
# There are currently no licenses. 
#
#--------------------------------------------------------------------------
# Description: 
#
#           Graph Laplacian Based Clustering         
#
#--------------------------------------------------------------------------

from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import numpy as np
import util
from util import Parameters
from util.build_graph import _graph_params_default_values
from util import misc
from util import BuildGraph
from sklearn.cluster import KMeans
reload(misc)
reload(util)


################################################################################
#######################Subroutines for Main Classifier##########################
################################################################################
def _diffusion_step_eig(v,V,E,dt):
    """diffusion on graphs
    """
    if len(v.shape) > 1:
        return np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1+dt*E)))
        return u_new.ravel()

def _gl_forward_step(u_old,dt,eps):
    v = u_old-dt/eps*(np.power(u_old,3)-u_old) #double well explicit step
    return v

def _l2_fidelity_gradient_binary(u_old,dt,fid,eta):
    temp = fid[:,1].ravel()
    v = u_old.copy()
    v[fid[:,0].astype(int).ravel()] = v[fid[:,0].astype(int).ravel()]+ dt*eta*(temp-v[fid[:,0].astype(int).ravel()]) # gradient step
    return v    

def _mbo_forward_step_binary(u): #thresholding
    v = u.copy()
    v[v<0] = -1
    v[v>0] = 1
    return v

def _l2_fidelity_gradient_multiclass(u_old,dt,fid_ind,fid_vec,eta):
    # temp = fid[:,1].ravel()
    # temp = util.labels_to_vector(temp) # convert to matrix form
    v = u_old.copy()
    v[fid_ind.astype(int).ravel(),:] = v[fid_ind.astype(int).ravel(),:]+ dt*eta*(fid_vec-v[fid_ind.astype(int).ravel(),:]) # gradient step
    return v    

def _mbo_forward_step_multiclass(u): #thresholding
    return util.labels_to_vector(util.vector_to_labels(u),vec_dim = u.shape[1])

def threshold(u, thre_val = 0):
    w = u.copy()
    w[w<thre_val] = 0
    w[w>thre_val] = 1
    return w


################################################################################
####################### Main Classifiers for Classification #####################
################################################################################


##### Binary Ginzburg with fidelity, using eigenvectors #######
def gl_binary_supervised_eig(V,E,fid,dt,u_init,eps = 1,eta = 1, tol = 1e-5,Maxiter = 500):
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eps : scalar, 
        diffuse interface parameter
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    u_init : ndarray, shape (n_samples ,1)
        initial u_0 for the iterations
    """

    #performing the Main GL iteration with fidelity
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = _gl_forward_step(u_old,dt,eps)
        v = _l2_fidelity_gradient_binary(v,dt,fid = fid, eta = eta)
        u_new = _diffusion_step_eig(v,V,E,eps*dt)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new


##### Binary Laplacian Smoothing, using eigenvectors #######
def lap_binary_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Laplacian Smoothing (Used for Benchmarking. Not actually in the LaplacianClustering class)
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        w = _l2_fidelity_gradient_binary(u_old,dt,fid = fid, eta = eta)
        u_new = _diffusion_step_eig(w,V,E,dt)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new 



##### Binary MBO with fidelity, using eigenvectors #######
def mbo_binary_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500,inner_step_count = 10): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    #performing the Main MBO iteration with fidelity
    i = 0
    u_new = u_init.copy()
    u_diff = 1
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_binary(v,dt,fid = fid, eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_binary(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new 


##### MBO Zero Means, using eigenvectors #######
def mbo_zero_means_eig(V,E,dt,u_init,tol = .5,Maxiter = 500,inner_step_count = 5): # inner stepcount is actually important! and can't be set to 1...
    """ The MBO scheme with a forced zero mean constraint. Valid only for binary classification. 
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        w = u_old.copy()
        for k in range(inner_step_count): # diffuse and threshold for a while
            v = _diffusion_step_eig(w,V,E,dt)
            w = v-np.mean(v) # force the 0 mean
        u_new = _mbo_forward_step_binary(w)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new

##### MBO Zero Means, using eigenvectors #######
def gl_zero_means_eig(V,E,dt,u_init,eps = 1, tol = 1e-5,Maxiter = 500, inner_step_count = 5): 
    """ The MBO scheme with a forced zero mean constraint. Valid only for binary classification. 
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        w = u_old.copy()
        for k in range(inner_step_count): # diffuse and threshold for a while
            v = _diffusion_step_eig(w,V,E,eps*dt)
            w = v-np.mean(v) # force the 0 mean
        u_new = _gl_forward_step(w,dt,eps)
        u_diff = (abs(u_new-u_old)).sum()

        i = i+1
    return u_new  

##### Multiclass MBO with fidelity, using eigenvectors #######
def mbo_multiclass_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500,inner_step_count = 10): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    u_init : ndarray, shape(n_samples, n_class)
        initial condition of scheme
    dt : float
        stepsize for scheme
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    #performing the Main MBO iteration with fidelity
    i = 0
    u_new = u_init.copy()
    u_diff = 1
    fid_ind = fid[:,0]
    fid_vec = util.labels_to_vector(fid[:,1])
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_multiclass(v,dt,fid_ind = fid_ind, fid_vec = fid_vec,eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_multiclass(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new    


##### MBO Modularity, Multiclass Next  #######
def mbo_modularity_eig(V,E,dt,u_init,k_weights,gamma = .5, tol = .5,Maxiter = 500,inner_step_count = 5): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    u_init : ndarray, shape(n_samples, n_class)
        initial condition of scheme
    dt : float
        stepsize for scheme
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    #performing the Main MBO iteration with fidelity
    i = 0
    if len(k_weights.shape) == 1:
        k_weights = k_weights.reshape(len(k_weights),1)
    # convert u_init to standard multiclass form for binary tags
    if (len(u_init.shape)== 1) or (u_init.shape[1] == 1): 
        u_init = misc.labels_to_vector(misc.to_standard_labels(u_init))
    u_new = u_init.copy()
    u_diff = 10
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        w = v.copy()
        for k in range(inner_step_count):
            graph_mean_v = np.dot(k_weights.T,v)/np.sum(k_weights)
            w += 2.*gamma*dt*k_weights*(v-graph_mean_v)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_multiclass(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new  






##### MBO Chan-Vese, Multiclass Next  #######






#######################################################################################
########################## The Main Class Definitions #################################
#######################################################################################

class LaplacianClustering(Parameters):
    """ Apply a Laplacian Graph-cut Solver(either MBO or Ginzburg-Landau) to solve
    a semi-supervised or unsupervised clustering problem. 
    semi-supervised minimizes approximately |u|_GraphTV + (u-f)^2, f being the fidelity 
    unsupervised minimizes approximately |u|_GraphTV + balancing term for cluster size
    currently only supports binary classifiation. 

    Class Overview:
    -----------    
        Attributes:    
        -- various scheme specific parameters(scheme_type, n_class...)
        -- self.graph : A BuildGraph Object.
            Containing graph params and the computed graph Laplacian. 
        -- self.data : A Parameter Object. 
            Containing the raw data, ground truth label 
            
        Methods : 
        -- Constructor : set scheme specific parameters. 
        -- build_graph : build the graph Laplacian, specifying the graph parameters
        -- load_data : load raw data into model.  
        -- generate_random_fidelity : generate some random fidelity
        -- fit_predict : predict labels for the data. 


    Class Constructor Parameters
    ----------------------------
    scheme_type : String {'GL_fidelity','MBO_fidelity','spectral_clustering',
                    GL_zero_means','MBO_zero_means','modularity'}
        Types of scheme for the classifiation. First two are for semi-supervised learning, and 
        last four for unsupervised learning.     
    n_class : integer 
        number of classes (This can be inferred from ground_truth if provided)   
    u_init : ndarray, shape(n_samples,)
        initial labels or score for algorithm
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    eta : scalar
        fidelity strength term
    eps : scalar
        diffuse interface parameter(only for Ginzburg-Landau schemes)
    dt : scalar
        Learning stepsize for the scheme
    inner_step_count : int
        for MBO, GL_zero_means, Modularity. Number of times the scheme performs
        "diffuse" + "forcing" before doing "threshold"

    Data Parameters(in self.data)
    ----------------------------    
    ( use load_data() to change values )
    raw_data : ndarray, (n_samples, n_features)
        raw data for the classification task
    ground_truth : ndarray, (n_samples,).{0...K-1} labels.
        labels corresponding to the raw data  

    Graph Parameters and Data(in self.graph)
    ----------------------------    
    ( use set_graph_params() to change values )
    See self.build_graph() for more details

    Other Class Attributes
    ---------------------- 
    laplacian_matrix_ : array-like, shape (n_samples, n_samples)
        graph laplacian matrix or a dictionary {'V':V,'E':'E'} containing eigenvectors and
        eigenvalues of the laplacian. 
    labels_ :
        Labels or the score of each point    
    """ 

    ## default values for relavant parameters. 
    _params_default_values = {'scheme_type': None, 'n_class': None, 'data': Parameters(), 
    'fid':None,'eps':None, 'eta':None, 'dt': None, 'u_init' : None,'inner_step_count' : None,
    'gamma': None} 

    def __init__(self, **kwargs): # look how clean the constructor is using inheritance! 
        self.set_parameters(**kwargs)
        self.set_to_default_parameters(self._params_default_values)
        self.graph = BuildGraph()

    def load_data(self, raw_data  = None, ground_truth = None):
        """
            raw_data : ndarray, (n_samples, n_features)
            ground_truth : ndarray, (n_samples,).{0...K-1} labels.
        """
        if not raw_data is None:
            self.nclass = None # reset number of classes
            self.graph = BuildGraph() # reset the graph every time new data is loaded 
            self.data.raw_data = raw_data
            if hasattr(self,'fid'):
                self.fid = None
        if not ground_truth is None:
            # infer the label from ground_truth if available. 
            self.nclass = None #reset the number of classes
            self.n_class = np.unique(ground_truth).shape[0] 
            if np.unique(ground_truth).shape[0] == 2 :# convert labels binary case
                self.data.ground_truth = util.to_binary_labels(ground_truth)
            else:
                self.data.ground_truth = util.to_standard_labels(ground_truth)
            if hasattr(self,'fid'): #reset fidelity after loading ground_truth
                self.fid = None       

    def set_graph_params(self,**kwargs):
        """ Available Parameters for Graphs

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
        """

        try : 
            self.graph.set_parameters(**kwargs)
        except : 
            raise AttributeError("self.graph Non-existent. Use .build_Laplacian() to construct the graph object")

    def build_graph(self, **kwargs): # build the graph Laplacian
        """ Construct and compute the graph Laplacian

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
        """
        if kwargs:
            if hasattr(self,'graph'):
                self.clear('graph')
            self.graph = BuildGraph(**kwargs)
        self.graph.build_Laplacian(self.data.raw_data)

    def generate_initial_value(self, opt = 'rd_equal'):
        # infer the label from ground_truth if available. 
        try : 
            if self.n_class is None:
                if not self.data.ground_truth is None:
                    self.n_class = np.unique(self.data.ground_truth).shape[0] 
                else:
                    raise AttributeError("No ground truth found. Need to specify n_class via set_parameters() ")
        except : 
            raise AttributeError("Either the data or the ground_truth is not specified. Cannot infer class number")
        if self.n_class == 2:
            if opt != 'eig':
                self.u_init = util.generate_initial_value_binary(opt = opt, V = None, n_samples = self.data.raw_data.shape[0])
                if 'modularity' in self.scheme_type: #the modularity method has inherently 0-1 vector labels
                    self.u_init = util.labels_to_vector(util.to_standard_labels(threshold(self.u_init)))
            else:
                try:
                    self.u_init = util.generate_initial_value_binary(opt = 'eig', V = self.graph.laplacian_matrix_['V'])
                except KeyError:
                    raise KeyError("laplacian_matrix_ needs to be in eigenvector format")
        elif opt != 'eig':
            self.u_init = util.generate_initial_value_multiclass(opt = opt, n_samples = self.data.raw_data.shape[0], n_class = self.n_class)
        else:
            raise NameError("Eig Option is currently unavailable for multiclass data")


    def generate_random_fidelity(self,percent = .05):
        try : 
            tags = np.unique(self.data.ground_truth)
        except AttributeError:
            raise AttributeError("Please provide ground truth")
        self.fid = np.zeros([0,2])
        for i, tag in enumerate(tags):
            ind_temp = util.generate_random_fidelity(ind =  np.where(self.data.ground_truth == tag)[0] , perc = percent)
            ind_temp = ind_temp.reshape(len(ind_temp), 1)
            tag_temp = tag*np.ones([len(ind_temp),1])
            fid_temp = np.concatenate((ind_temp, tag_temp), axis = 1)
            self.fid = np.concatenate((self.fid,fid_temp), axis = 0)


    def fit_predict(self):
        # build the laplacian if there is non existent
        if hasattr(self,'labels_'):
            self.clear('labels_')
        try:
            if self.graph.laplacian_matrix_ is None:
                raise AttributeError("Build The Graph using build_graph() before calling fit_predict")
        except AttributeError:
            raise AttributeError("Build The Graph using build_graph() before calling fit_predict")
        # infer the label from ground_truth if available. 
        try : 
            if self.n_class is None:
                if not self.data.ground_truth is None:
                    self.n_class = np.unique(self.data.ground_truth).shape[0] 
                else:
                    raise AttributeError("No ground truth found. Need to specify n_class via set_parameters() ")
        except : 
            raise AttributeError("Either the data or the ground_truth is not specified. Cannot infer class number")

        if self.scheme_type.find("fidelity") != -1:
            if self.fid is None:
                print("Fidelity point not provided. Generating 5 percent random fidelity.")
                self.generate_random_fidelity()

        if self.u_init is None:
            print("u_init not provided. Generating random initial condition.")
            self.generate_initial_value()

        # check if the laplacian is in eigenvector form
        if type(self.graph.laplacian_matrix_) is dict:
            V = self.graph.laplacian_matrix_['V']
            E = self.graph.laplacian_matrix_['E']

        # wrapper to check which scheme to use.     
        if self.scheme_type == 'GL_fidelity':
            inner_step_count = None
            if self.eta is None:
                self.eta = 1
                print("Warning, fidelity strength eta not supplied. Using default value 1")      
            if self.dt is None:
                self.dt = .5
                print("Warning, stepsize dt not supplied. Using default value .1")   
            if self.eps is None:
                self.eps = 1
                print("Warning, scale interface parameter eps not supplied. Using default value 1")                     
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2 :
                    labels = gl_binary_supervised_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],fid = self.fid ,dt = self.dt, u_init = self.u_init ,
                        eps = self.eps ,eta = self.eta)
                    self.soft_labels_ = labels
                    labels[labels<0] = -1
                    labels[labels>0] = 1
                    self.labels_ = labels 
                else:
                    raise ValueError("Ginzburg-Landau Schemes only for 2 class segmentation")
                    return
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph")
        elif self.scheme_type == 'MBO_fidelity':   
            inner_step_count = None     
            if self.eta is None:
                self.eta = 1
                print("Warning, fidelity strength eta not supplied. Using default value 1")      
            if self.dt is None:
                self.dt = 1
                print("Warning, stepsize dt not supplied. Using default value 1")                
            if self.inner_step_count is None:
                inner_step_count = 10 # default value
            else : 
                inner_step_count = self.inner_step_count
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    self.labels_ = mbo_binary_supervised_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],fid = self.fid 
                        ,dt = self.dt, u_init = self.u_init ,eta = self.eta, inner_step_count = inner_step_count)
                else:
                    res = mbo_multiclass_supervised_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],fid = self.fid ,dt = self.dt, 
                        u_init = self.u_init ,eta = self.eta) 
                    self.labels_ = util.vector_to_labels(res)                  
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph")    
        elif self.scheme_type == 'MBO_zero_means':    
            if self.dt is None:
                self.dt = .1
                print("Warning, stepsize dt not supplied. Using default value .1")   
            if self.inner_step_count is None:
                inner_step_count = 5 # default value
            else : 
                inner_step_count = self.inner_step_count                
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    self.labels_ = mbo_zero_means_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],dt = self.dt
                        , u_init = self.u_init, inner_step_count = inner_step_count)                 
                else:
                    raise ValueError("MBO_zero_means Schemes only for 2 class segmentation")              
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph") 
        elif self.scheme_type == 'GL_zero_means': 
            inner_step_count = None  
            if self.dt is None:
                self.dt = .1
                print("Warning, stepsize dt not supplied. Using default value .1")       
            if self.inner_step_count is None:
                inner_step_count = 10 # default value
            else : 
                inner_step_count = self.inner_step_count  
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    labels = gl_zero_means_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],eps = self.eps,dt = self.dt
                        , u_init = self.u_init , inner_step_count = inner_step_count)
                    self.soft_labels_ = labels
                    labels[labels<0] = -1
                    labels[labels>0] = 1
                    self.labels_ = labels                     
                else:
                    raise ValueError("Ginzburg-Landau Schemes only for 2 class segmentation")              
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph") 

        elif self.scheme_type == 'MBO_modularity':
            inner_step_count = None
            temp = np.ones([self.data.raw_data.shape[0],1])
            if self.inner_step_count is None:
                inner_step_count = 10 # default value
            else : 
                inner_step_count = self.inner_step_count             
            if self.gamma is None:
                self.gamma = 1
                print("Warning, Modularity parameter gamma not supplied. Using default value 1")      
            if self.dt is None:
                self.dt = .1
                print("Warning, stepsize dt not supplied. Using default value .1")                   
            if type(self.graph.laplacian_matrix_) is dict:
                res = mbo_modularity_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],k_weights = temp, 
                    dt = self.dt, u_init = self.u_init, gamma = self.gamma ,inner_step_count = inner_step_count)
                self.labels_ = util.vector_to_labels(res)                 
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph")

        elif self.scheme_type == 'spectral_clustering': # added for benchmark  
        # this just performs k-means after doing spectral projection                
            if type(self.graph.laplacian_matrix_) is dict:
                cf = KMeans(n_clusters = self.n_class)
                temp = self.graph.laplacian_matrix_['V'][:,1:]
                self.labels_ = cf.fit_predict(temp)              
            else : 
                raise AttributeError("Spectral Clustering only supported for laplacian_matrix_ in eigenvector format")

    def compute_error_rate(self):
        try : 
            if (self.data.ground_truth is None):
                raise ValueError("Please provide ground truth labels when using compute_error_rate() ")
        except : 
            raise ValueError("Please provide ground truth labels when using compute_error_rate() ")       
        self.error_rate_ = util.compute_error_rate(ground_truth = self.data.ground_truth, labels = self.labels_)
        return self.error_rate_






        

