""" script for generating three moons"""
import numpy as np 
import numpy.random as rd 
from numpy.random import permutation

def three_moons(n_samples=600, noise=0.03,shuffle = False):
    """ Generate Three Moons dataset 
    Parameters
    -----------
    n_samples : int
        number of samples for three_moons
    noise : float
        standard variance of the moons
    shuffle : bool
        if true, shuffle the return. 
    Return
    -------
    X : ndarray, shape (n_samples, 2)
        actual datapoints of X
    Y : labels for X. Uses 0,1,2,3,4... for class labels   
    """
    n_samples = int(np.floor(n_samples/3.)*3) 
    X = np.zeros([n_samples,2])
    Y = np.zeros(n_samples)
    thetas = np.linspace(0,np.pi,n_samples/3)
    r = .5
    h = .25
    dx = .6
    # construct first moon 
    X[:n_samples/3,:] = np.array([ [r*np.cos(theta)+r, r*np.sin(theta)] for theta in thetas ])
    Y[:n_samples/3] = np.zeros(n_samples/3)
    # construct second moon
    X[n_samples/3:n_samples*2/3,:] = np.array([ [r*np.cos(-theta)+r+dx, r*np.sin(-theta)+h] for theta in thetas ])
    Y[n_samples/3:n_samples*2/3]= np.ones(n_samples/3)
    # construct third moon
    X[n_samples*2/3:,:] = np.array([ [r*np.cos(theta)+r+2*dx, r*np.sin(theta)] for theta in thetas ])
    Y[n_samples*2/3:]= 2*np.ones(n_samples/3)

    # add some noise
    X += rd.randn(n_samples,2)*noise

    #shuffle if needed
    if(shuffle):
        ind = permutation(n_samples)
        X = X[ind,:]
        Y = Y[ind,:]
    return X,Y
