from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import numpy as np
import util
from util import Parameters
from util.build_graph import _graph_params_default_values
from util import misc
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
    return util.labels_to_vector(util.vector_to_labels(u))

def threshold(u, thre_val = 0):
    w = u.copy()
    w[w<thre_val] = 0
    w[w>thre_val] = 1
    return w


################################################################################
####################### Main Classifiers for Classification #####################
################################################################################


##### Binary Ginzburg with fidelity, using eigenvectors #######
def gl_binary_supervised_eig(V,E,fid,dt,u_init,eps = 1,eta = 1, tol = 1e-5,Maxiter = 1000):
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
        u_diff = (abs(u_new-u_old)).max()
        i = i+1
    return u_new


##### Binary Laplacian Smoothing, using eigenvectors #######
def lap_binary_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Laplacian Smoothing 
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
        u_diff = (abs(u_new-u_old)).max()
        i = i+1
    return u_new 



##### Binary MBO with fidelity, using eigenvectors #######
def mbo_binary_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500,inner_step_count = 15): # inner stepcount is actually important! and can't be set to 1...
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
    fid_ind = fid[:,0]
    fid_vec = labels_to_vector(fid[:,1])
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_binary(v,dt, eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_binary(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new 


##### MBO Zero Means, using eigenvectors #######
def mbo_zero_means_eig(V,E,dt,u_init,tol = .5,Maxiter = 500,inner_step_count = 15): # inner stepcount is actually important! and can't be set to 1...
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
    return w 

##### MBO Zero Means, using eigenvectors #######
def gl_zero_means_eig(V,E,dt,u_init,eps = 1, tol = 1e-5,Maxiter = 1000, inner_step_count = 15): 
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
            w = w-np.mean(v) # force the 0 mean
        u_new = _gl_forward_step(w,dt,eps)
        u_diff = (abs(u_new-u_old)).max()
        i = i+1
    return u_new  

##### Multiclass MBO with fidelity, using eigenvectors #######
def mbo_multiclass_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500,inner_step_count = 15): # inner stepcount is actually important! and can't be set to 1...
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
    print i
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
    Parameters
    -----------
    scheme_type : {'GL_fidelity','MBO_fidelity','GL_zero_means','MBO_zero_means','MBO_modularity'}
        Types of scheme for the classifiation. First two are for semi-supervised learning, and 
        last three for unsupervised learning. 
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.
    n_channels : int
        Number of channels, used for image data.         
    reg_params : dictionary
        dictionary for the regularization parameters of the model
    u_init : ndarray, shape(n_samples,)
        initial labels or score for algorithm
    affinity : string, array-like or callable
        affinity matrix specification. If a string, this may be one 
        of 'nearest_neighbors','rbf'. 
    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``
    eigen_solver : {None, 'arpack', 'nystrom'}
        Solver used for computing eigenvectors. Only rbf is supported for nystrom. 
    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    ground_truth : ndarray shape(n_samples,)
        user supplied ground truth with labels. Used for generating fidelity term
    Neig : integer
        number of eigenvectors to compute.
    Attributes
    ----------
    laplacian_matrix_ : array-like, shape (n_samples, n_samples)
        graph laplacian matrix or a dictionary {'V':V,'E':'E'} containing eigenvectors and
        eigenvalues of the laplacian. 
    labels_ :
        Labels or the score of each point
    References
    ----------

    """ 

    ## default values for relavant parameters. 
    _params_default_values = {'scheme_type': None, 'n_class': 2, 'laplacian_matrix_': None ,
    'iter_params':Parameters(), 'graph_params': Parameters(), 'data':Parameters(), 'ground_truth':None, 
    'fid':None, 'graph_deg':None}

    _iter_params_default_values = {'eps':1., 'eta':1., 'dt': None , 'gamma':1.}     

    def __init__(self, **kwargs): # look how clean the constructor is using inheritance! 
        Parameters.__init__(self,**kwargs)
        self.set_to_default_parameters(self._params_default_values)
        self.iter_params.set_to_default_parameters(self._iter_params_default_values)

    def load_raw_data(self, raw_data  = None, ground_truth = None, u_init = None, fid = None):
        if not u_init is None:
            self.u_init = u_init
        if not raw_data is None:
            if len(raw_data.shape) == 3:
                self.data.n_channels = raw_data.shape[2]
                self.data.raw_data = util.flatten_23(raw_data)
            else:
                self.data.raw_data = raw_data
                self.data.n_channels = 1
        if not ground_truth is None:
            self.ground_truth = ground_truth
        if not fid is None:
            self.fid = fid
        if not raw_data is None:
            self.laplacian_matrix_ = None # reset the laplacian matrix every time new data is loaded.  

    # def set_parameters(self, scheme_type = None, iter_params = {}, n_class = None, clear_params = False):
    #     if clear_params:
    #         self.iter_params.clear()
    #     if scheme_type != None:
    #         self.scheme_type = scheme_type
    #     if n_class != None: 
    #         self.n_class = n_class
    #     if iter_params :
    #         self.iter_params.set(iter_params) ## depreciated, all parameter settings goes to the parameter class        

    def set_graph_parameters(self, **kwargs): #interface for specifically setting graph parameters
        self.graph_params.set_parameters(**kwargs)
        self.graph_params.set_to_default_parameters(_graph_params_default_values)


    def build_Laplacian(self):
        if self.graph_params.Eig_solver == 'nystrom': # add code for Nystrom Extension separately             
            E,V, graph_deg = util.nystrom(raw_data = self.data.raw_data,n_channels = self.data.n_channels, graph_params = self.graph_params )
            E = E[:self.graph_params.Neig]
            V = V[:,:self.graph_params.Neig]
            self.laplacian_matrix_ = {'V': V, 'E': E}
            self.graph_deg = graph_deg
        else: 
            Lap,graph_deg = util.build_laplacian_matrix(raw_data = self.data.raw_data,graph_params = self.graph_params)
            self.graph_deg = graph_deg
            if self.graph_params.Eig_solver  == 'arpack':
                E,V = util.generate_eigenvectors(Lap,self.graph_params.Neig)
                E = E[:,np.newaxis]
                self.laplacian_matrix_ = {'V': V, 'E': E}
                return 
            elif self.graph_params.Eig_solver == 'full':
                self.laplacian_matrix_ = Lap
                return

    def generate_initial_value(self, opt = 'rd_equal'):
        if self.n_class == 2:
            if opt != 'eig':
                self.u_init = util.generate_initial_value_binary(opt = opt, V = None, n_samples = self.data.raw_data.shape[0])
                if 'modularity' in self.scheme_type:
                    self.u_init = util.labels_to_vector(util.to_standard_labels(threshold(self.u_init)))
            else:
                self.u_init = util.generate_initial_value_binary(opt = 'eig', V = self.laplacian_matrix_['V'])
        elif opt != 'eig':
            self.u_init = util.generate_initial_value_multiclass(opt = opt, n_samples = self.data.raw_data.shape[0], n_class = self.n_class)



    def generate_random_fidelity(self,percent = .05):
        tags = np.unique(self.ground_truth)
        self.fid = np.zeros([0,2])
        for i, tag in enumerate(tags):
            ind_temp = util.generate_random_fidelity(ind =  np.where(self.ground_truth == tag)[0] , perc = percent)
            ind_temp = ind_temp.reshape(len(ind_temp), 1)
            tag_temp = tag*np.ones([len(ind_temp),1])
            fid_temp = np.concatenate((ind_temp, tag_temp), axis = 1)
            self.fid = np.concatenate((self.fid,fid_temp), axis = 0)

    def get_graph_deg(self):
        return self.graph_deg

    def fit_predict(self):
        # build the laplacian if there is non existent
        if self.laplacian_matrix_ ==  None:
            self.build_Laplacian()

        # check if the laplacian is in eigenvector form
        if type(self.laplacian_matrix_) is dict:
            V = self.laplacian_matrix_['V']
            E = self.laplacian_matrix_['E']

        # wrapper to check which scheme to use.     
        if self.scheme_type == 'GL_fidelity':
            eps = (1. if (self.iter_params.eps == None) else self.iter_params.eps )           
            dt = (eps/10. if (self.iter_params.dt == None) else self.iter_params.dt)
            eta = (1. if (self.iter_params.eta == None) else self.iter_params.eta )               
            if type(self.laplacian_matrix_) is dict:
                labels = gl_binary_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eps = eps ,eta = eta)
                self.soft_labels_ = labels
                labels[labels<0] = -1
                labels[labels>0] = 1
                self.labels_ = labels 
            else:
                pass
        elif self.scheme_type == 'MBO_fidelity':        
            eta = (1. if (self.iter_params.eta == None) else self.iter_params.eta )  
            dt = (eta/5. if (self.iter_params.dt == None) else self.iter_params.dt )             

            if type(self.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    foo = self.fid.copy() #convert to binary labels! (this is frustrating but necessary)
                    foo[:,1] = util.standard_to_binary_labels(foo[:,1])
                    self.labels_ = mbo_binary_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = foo ,dt = dt, u_init = self.u_init ,
                    eta = eta)
                else:
                    res = mbo_multiclass_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eta = eta) 
                    self.labels_ = util.vector_to_labels(res)                  
            else:
                pass
        elif self.scheme_type == 'Lap_fidelity':
            eta = (1. if (self.iter_params.eta == None) else self.iter_params.eta )  
            dt = (eta/5. if (self.iter_params.dt == None) else self.iter_params.dt ) 

            if type(self.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    labels = lap_binary_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eta = eta)
                    self.soft_labels_ = labels
                    labels[labels<0] = -1
                    labels[labels>0] = 1
                    self.labels_ = labels                     
                else:
                    res = mbo_multiclass_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eta = eta) 
                    self.labels_ = util.vector_to_labels(res)                  
            else:
                pass        
        elif self.scheme_type == 'MBO_zero_means':
            dt = (2. if (self.iter_params.dt == None) else self.iter_params.dt ) 
            if type(self.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    self.labels_ = mbo_zero_means_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],dt = dt, u_init = self.u_init)                 
                else:
                    print "Only supports Binary classifiation"
                    return                    
            else:
                pass 
        elif self.scheme_type == 'GL_zero_means':
            eps = (1. if (self.iter_params.eps == None) else self.iter_params.eps )           
            dt = (eps/5. if (self.iter_params.dt == None) else self.iter_params.dt)

            if type(self.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    labels = gl_zero_means_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],eps = eps,dt = dt, u_init = self.u_init )
                    self.soft_labels_ = labels
                    labels[labels<0] = -1
                    labels[labels>0] = 1
                    self.labels_ = labels                     
                else:
                    print "Only supports Binary classifiation"
                    return                 
            else:
                pass   

        elif self.scheme_type == 'MBO_modularity':
            if(self.graph_params.Laplacian_type != 'u'):
                print "Warning: The algorithm assumes the use of unnormalized Laplacian"
                temp = np.ones([self.data.raw_data.shape[0],1])
            else:
                temp = self.graph_deg
            gamma = (1. if (self.iter_params.gamma == None) else self.iter_params.gamma )           
            dt = (1. if (self.iter_params.dt == None) else self.iter_params.dt)
            if type(self.laplacian_matrix_) is dict:
                res = mbo_modularity_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],k_weights = temp, dt = dt, u_init = self.u_init, gamma = gamma )
                self.labels_ = util.vector_to_labels(res)                 
            else:
                pass  


    def compute_error_rate(self):
        self.error_rate_ = util.compute_error_rate(ground_truth = self.ground_truth, labels = self.labels_)
        return self.error_rate_






        

