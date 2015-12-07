import os, struct
from array import array
import numpy as np
from numpy.random import permutation
def read_mnist(digits, path = "."):
    """
    Python function for importing the MNIST data set.
    """

    fname_img = os.path.join(path, 'data/train-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 'data/train-labels-idx1-ubyte')

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.array(array("b", flbl.read()))
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.array(array("B", fimg.read()))
    fimg.close()
    img = img.reshape(size,rows*cols)
    ind = np.array([ k for k in xrange(size) if lbl[k] in digits ]).astype(int)

    images = img[ind,:]
    labels = lbl[ind]

    return images, labels

def subsample(sample_num, labels = None, **kwargs):
    res = ()
    if not labels is None: #return amount of sample from each label
        num_tags = len(np.unique(labels))
        reslable = None
        sm = sample_num
        for i, argname in enumerate(kwargs):
            temp = None
            val = kwargs[argname]
            for j,tag in enumerate(np.unique(labels)):
                val = kwargs[argname]
                p = permutation(sm)
                if temp is None:
                    temp = val[labels == tag][p]
                else:
                    temp = np.concatenate((temp,val[labels == tag][p]), axis = 0)
            res = res + (temp,)
        for j, tag in enumerate(np.unique(labels)):
            p = permutation(sm)
            if reslable is None:
                reslable = labels[labels == tag][p]
            else:
                reslable = np.concatenate((reslable,labels[labels == tag][p]), axis = 0)
        return res + (reslable,) 
    else: #just sample some points
        sm = sample_num
        for i, argname in enumerate(kwargs):
            val = kwargs[argname]
            p = permutation(sm)
            res = res + (val[p],)   
        return res
