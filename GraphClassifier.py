import numpy as np 
from numpy import random





def diffusion_step(v,V,E,Neig,dt,Nstep = 1):
	"""diffusion on graphs
	"""
	u_new = V*np.divide(V.T*v,np.power(1+dt*E.T,Nstep))
	return u_new

def gl_forward_step(u_old,dt,eps,fid = None, eta = None):
	v = u_old-dt/eps*(np.power(u_old,3)-u_old) #double well explicit step
	if fid != None:
		temp = fid[1][np.newaxis].T
		v[fid[0]] = v[fid[0]]+ dt*eta*(temp-v[fid[0]])
	return v


#####Ginzburg Landau Classification #######
def classify_binary_supervised(V,E,fid,dt,uo,eps = 1,eta = 1, tol = 1e-6,Maxiter = 1000):
	""" Ginzburg Landau with fidelity
	V,E are the eiven values and eigenvector of the graphs
	dt is the stepsize
	eps is the scale parameter epsilon
	eta is the scalar in front of fidelity
	fid is the fidelity set
	tol, Maxiter are the stopping criterions of the iteration
	"""
	N = V.shape[0]
	Neig = V.shape[1]

	#performing the Main GL iteration with fidelity
	i = 0
	u_new = uo.copy()
	u_diff = 1

	while (i<Maxiter) and (u_diff > tol):
		u_old = u_new.copy()
		v = gl_forward_step(u_old,dt,eps,fid,eta = eta)
		u_new = diffusion_step(v,V,E,Neig,eps*dt)
		u_diff = (abs(u_new-u_old)).max()
		i = i+1
	return u_new

def classify_zero_means(V,E,dt,uo,eps=1,tol = 1e-6,Maxiter = 1000):
	""" Ginzburg Landau with zero means
	enforce 0-mass constraint at every iteration
	"""
	N = V.shape[0]
	Neig = V.shape[1]

	#performing the Main GL iteration with zero means
	i = 0
	u_new = uo.copy()
	u_diff = 1

	while (i<Maxiter) and (u_diff > tol):
		u_old = u_new.copy()
		v = gl_forward_step(u_old,dt,eps) #double well explicit step
		u_new = diffusion_step(v,V,E,Neig,eps*dt)
		u_new = u_new - np.mean(u_new)
		u_diff = (abs(u_new-u_old)).max()
		i = i+1
	return u_new



###########MBO Classification ####################
def mbo_zero_means(V,E,dt,uo,tol = 1e-6,Maxiter = 200,Nstep = 3):
	"""MBO scheme with 0 mass constraint
	Nstep is the number of diffusion steps 
	"""
	N = V.shape[0]
	Neig = V.shape[1]

	#performing the Main GL iteration with zero means
	i = 0
	u_new = uo.copy()
	u_diff = 2

	while (i<Maxiter) and (u_diff > tol):
		u_old = u_new.copy()
		u_new = np.sign(u_new)
		u_new = diffusion_step(u_new,V,E,Neig,dt,Nstep)
		u_new = u_new - np.mean(u_new)
		u_diff = (abs(u_new-u_old)).sum()
		i = i+1
	return u_new




class GL_Classifier:
	"""GL classifier class
	takes the graph information and basic parameters during initialization 
	to avoid proliferation of function arguments
	"""
	def __init__(self,dt,eps = 1,eta = 1,V = None, E = None, uo = None):
		self.dt = dt
		self.eps = eps
		self.eta = eta
		self.V = np.asmatrix(V)
		self.E = np.asmatrix(E)
		self.N = self.V.shape[0]
		self.Neig = self.V.shape[1]
		self.eps = eps
		self.eta = eta

		#initializing according to different modes
		N = self.N
		if uo != None:
			if isinstance(uo,str):
				if uo == 'zero':
					self.uo = np.zeros(N,1)
				elif uo == 'random':
					self.uo = np.random.rand(N,1)*2-1
				elif uo == 'eig':
					self.uo = V[:,1]
			else:
				self.uo = uo.reshape(N,1)
		else:
			self.uo = np.zeros([N,1])
		self.uo = np.asmatrix(self.uo).reshape(N,1)

	def classify_binary_supervised(self,fid,tol = 1e-6,Maxiter = 1000):
		return classify_binary_supervised(V = self.V,E = self.E,fid = fid,dt = self.dt,uo = self.uo,eps = self.eps,eta = self.eta, tol = tol ,Maxiter = Maxiter )

	def classify_zero_means(self,tol = 1e-6,Maxiter = 1000):
		return classify_zero_means(V = self.V,E = self.E,dt = self.dt,uo = self.uo,eps = self.eps ,tol = tol,Maxiter = Maxiter)


class MBO_Classifier:
	"""MBO classifier class
	takes the graph information and basic parameters during initialization 
	to avoid proliferation of function arguments
	"""
	def __init__(self,dt,V = None, E = None, uo = None):
		self.dt = dt
		self.V = np.asmatrix(V)
		self.E = np.asmatrix(E)
		self.N = self.V.shape[0]
		self.Neig = self.V.shape[1]

		#initializing according to different modes
		N = self.N
		if uo != None:
			if isinstance(uo,str):
				if uo == 'zero':
					self.uo = np.zeros(N,1)
				elif uo == 'random':
					self.uo = np.random.rand(N,1)*2-1
				elif uo == 'eig':
					self.uo = V[:,1]
			else:
				self.uo = uo.reshape(N,1)
		else:
			self.uo = np.zeros([N,1])
		self.uo = np.asmatrix(self.uo).reshape(N,1)

	def mbo_zero_means(self,tol = 1,Maxiter = 200,Nstep = 3):
		return mbo_zero_means(V = self.V,E = self.E ,dt = self.dt,uo = self.uo,tol = 1e-6,Maxiter = 200,Nstep = 3)











