'Script for Generating The Data Sets of Two Moons'
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from GraphWeights import GraphWeights


def threshold_to_format(u):
    """Performs a sign threshold and returns a np 1D array of intergers 
    that are -1,1 valued 
    """
    labels = np.asarray(u) # If u comes as a matrix class, convert to array 
    labels = labels.flatten()  #flatten array to 1D
    labels = np.sign(labels) #thresholding 
    labels = labels.astype(int) #converting to int
    return labels


class TwoMoons(GraphWeights):
    """Generation, plotting of the Two Moons Dataset
    Has graph Laplacian and affinity matrix computation functionalities
    inherited from GraphWeights class
    """
    def __init__(self,N1,N2,Ndim = 100,sigma = np.sqrt(0.02)):
        self.N1 = N1 #number of points in class 1
        self.N2 = N2 #number of points in class 2
        self.sigma = sigma #standard deviation of Gaussian noise
        self.Ndim = Ndim #dimension of the Gaussian noise

    def gen_data(self):
        """ generate the two Moons data set  
        """
        self.X = np.zeros([self.N1+self.N2,self.Ndim])
        N1 = self.N1
        N2 = self.N2
        for i in xrange(0,N1-1):
            theta = np.pi*i/(N1-1)
            self.X[i,0:2] = [np.cos(theta),np.sin(theta)]
        for i in xrange(N1,N1+N2-1):
            theta = np.pi*(i-N1)/(N2-1)
            self.X[i,0:2] = [np.cos(-theta)+1,np.sin(-theta)+.5]
        self.X = self.X + rd.normal(size = [N1+N2,self.Ndim],scale = self.sigma)

    def gen_fidelity(self,Nfid1,Nfid2):
        """generate random fidelity consisting of a tuple(fid) of fidelity 
        fid[0] is the positions of the fidelity set
        fid[1] is the actual class assignment(+-1) of the fidelity set.
        """
        a = rd.randint(0,self.N1,Nfid1)
        a_val = np.ones(a.shape)
        b = rd.randint(self.N1,self.N1+self.N2,Nfid2)
        b_val = -np.ones(b.shape)
        c = np.concatenate((a,b),axis = 0)
        c_val = np.concatenate((a_val,b_val),axis = 0)
        return (c,c_val)

    def classification_error(self,u):
        """computes classification err of a given input u
        """
        N1 = self.N1
        N2 = self.N2
        labels = threshold_to_format(u)
        ground_truth = np.concatenate((np.ones(N1),-np.ones(N2)))
        temp = labels.dot(ground_truth) # Correct classification - Incorrect classification
        err = .5- temp/(2*(N1+N2))
        if err>0.5:
            err = 1-err
        return err


    def plot_data(self,mode = 'None', u = None,savefilename = None, titlename = None, show = False ):
        """A handy function for plotting and saving your segmentation results 
        could be expanded to have more functionalities 
        """

        if mode == 'None':
            plt.plot(self.X[0,:],self.X[1,:], 'ro')

        elif mode == 'Func':
            labels = threshold_to_format(u)
            colors = ['r' if label == 1 else 'b' for label in labels]
            plt.scatter(self.X[:,0], self.X[:,1], color = colors)
            plt.axis([-2,3,-1,1.5])
        if titlename != None:
            plt.title(titlename)
        if savefilename != None:
            plt.savefig(savefilename)
        if show == True:
            plt.show()








