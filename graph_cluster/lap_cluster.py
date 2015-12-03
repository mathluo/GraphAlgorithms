from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import numpy as np
import util
reload(util)





# define module functions here
def _diffusion_step_eig(v,V,E,dt):
    """diffusion on graphs
    """
    u_new = np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
    return u_new 

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

def _l2_fidelity_gradient_multiclass(u_old,dt,fid,eta):
    temp = fid[:,1].ravel()
    temp = util.labels_to_vector(temp) # convert to matrix form
    v = u_old.copy()
    v[fid[:,0].astype(int).ravel(),:] = v[fid[:,0].astype(int).ravel(),:]+ dt*eta*(temp-v[fid[:,0].astype(int).ravel(),:]) # gradient step
    return v    

def _mbo_forward_step_multiclass(u): #thresholding
    return util.labels_to_vector(util.vector_to_labels(u))



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
        v = _gl_forward_step_binary(u_old,dt,eps)
        v = _l2_fidelity_gradient_binary(v,dt,fid = fid, eta = eta)
        u_new = _diffusion_step_eig(v,V,E,eps*dt)
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

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_binary(v,dt,fid = fid, eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_binary(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return 

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

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_multiclass(v,dt,fid = fid, eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_multiclass(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new    


class LaplacianClustering:
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
        of 'nearest_neighbors', 'precomputed','rbf'. 
    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``
    precomputed_laplacian : ndarray, shape(n_samples, n_samples)
        precomputed graph laplacian, ignored if affinity != 'precomputed'
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
    def __init__(self, scheme_type, params = None, n_class = 2):
        if 'GL' in scheme_type:
            if 'eps' in params:
                self.eps = params['eps']
            else:
                self.eps = 1.
        if 'fid' in scheme_type:
            if 'eta' in params:
                self.eta = params['eta']
            else:
                self.eta = 1.
        self.scheme_type = scheme_type
        self.n_class = n_class
        self.laplacian_matrix_ = None


    def load_raw_data(self, raw_data  = None, ground_truth = None, u_init = None, fid = None):
        self.u_init = u_init
        if len(raw_data.shape) == 3:
            self.n_channels = raw_data.shape[2]
            raw_data = util.flat23(raw_data)
        else:
            self.raw_data = raw_data
            self.n_channels = 1
        self.ground_truth = ground_truth
        self.fid = fid
        self.laplacian_matrix_ = None # reset the laplacian matrix every time new data is loaded.  

    def set_parameters(self, scheme_type = None, params = None, n_class = 2):
        if scheme_type != None:
            self.scheme_type = scheme_type
        if n_class != None: 
            self.n_class = n_class
        if params!= None:
            if 'GL' in scheme_type:
                if 'eps' in params:
                    self.eps = params['eps']
                else:
                    self.eps = 1.
            if 'fid' in scheme_type:
                if 'eta' in params:
                    self.eta = params['eta']
                else:
                    self.eta = 1.            

    def set_graph_parameters(self, affinity = 'rbf', n_neighbors = None,  kernel_params = None, 
        Neig = None, fid = None,  Eig_solver = 'arpack', precomputed_laplacian = None):
        self.Neig = Neig
        self.Eig_solver = Eig_solver
        self.affinity = affinity
        self.kernel_params = kernel_params
        self.precomputed_laplacian = precomputed_laplacian
        self.n_neighbors = n_neighbors
        self.laplacian_matrix_ = None # reset the laplacian

    def build_Laplacian(self):
        if self.Eig_solver == 'nystrom': # add code for Nystrom Extension separately
            pass
        else: 
            Lap = util.build_laplacian_matrix(raw_data = self.raw_data,affinity = self.affinity, n_neighbors = self.n_neighbors)
            if self.Eig_solver  == 'arpack':
                E,V = util.generate_eigenvectors(Lap,self.Neig)
                E = E[:,np.newaxis]
                self.laplacian_matrix_ = {'V': V, 'E': E}
                return 
            elif self.Eig_solver == 'full':
                self.laplacian_matrix_ = Lap
                return

    def generate_initial_value(self, opt = 'rd_equal'):
        if self.n_class == 2:
            if opt != 'eig':
                self.u_init = util.generate_initial_value_binary(opt = opt, V = None, n_samples = self.raw_data.shape[0])
            else:
                self.u_init = util.generate_initial_value_binary(opt = 'eig', V = self.laplacian_matrix_['V'])
        else:
            if opt!= 'eig':
                self.u_init = util.generate_initial_value_multiclass(opt = opt, n_samples = self.raw_data.shape[0], n_class = self.n_class)


    def generate_random_fidelity(self,percent = .05):
        tags = np.unique(self.ground_truth)
        self.fid = np.zeros([0,2])
        for i, tag in enumerate(tags):
            ind_temp = util.generate_random_fidelity(ind =  np.where(self.ground_truth == tag)[0] , perc = percent)
            ind_temp = ind_temp.reshape(len(ind_temp), 1)
            tag_temp = tag*np.ones([len(ind_temp),1])
            fid_temp = np.concatenate((ind_temp, tag_temp), axis = 1)
            self.fid = np.concatenate((self.fid,fid_temp), axis = 0)

    def fit_predict(self):
        if self.laplacian_matrix_ ==  None:
            self.build_Laplacian()
        if type(self.laplacian_matrix_) is dict:
            V = self.laplacian_matrix_['V']
            E = self.laplacian_matrix_['E']
        if self.scheme_type == 'GL_fidelity':
            dt = self.eps/10.
            if type(self.laplacian_matrix_) is dict:
                self.labels_ = gl_binary_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eps = self.eps ,eta = self.eta)
            else:
                pass
        elif self.scheme_type == 'MBO_fidelity':
            dt = .6
            if type(self.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    self.labels_ = mbo_binary_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eta = self.eta)
                else:
                    res = mbo_multiclass_supervised_eig(self.laplacian_matrix_['V'],self.laplacian_matrix_['E'],fid = self.fid ,dt = dt, u_init = self.u_init ,
                    eta = self.eta) 
                    self.labels_ = util.vector_to_labels(res)                  
            else:
                pass
        elif self.scheme_type == 'MBO_modularity': 


    def compute_score(self):
        self.score_ = util.compute_error_rate(ground_truth = self.ground_truth, labels = self.labels_)
        return self.score_






        

