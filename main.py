"""Testing the usage of the Graph Algorithm Modules
Experiment: Test three graph classification methods 
compare plots and error rates
"""


from TwoMoons import TwoMoons 
from GraphClassifier import GL_Classifier
from GraphClassifier import MBO_Classifier
import numpy as np


print "Generation of Two Moons Data"
data = TwoMoons(1000,1000)
data.gen_data()

Nfid = 50
print "Generation of %d Fidelity Points" %Nfid
fid = data.gen_fidelity(Nfid,Nfid)


print "Computing eigenvectors"
E,V = data.smallest_eigenvv(40)



print "Classifying"
c1 = GL_Classifier(dt = 0.5, V = V, E = E, eps = 1, eta = 1,uo = 'random')
u_res_1 = c1.classify_binary_supervised(fid = fid)
u_res_2 = c1.classify_zero_means()

c2 = MBO_Classifier(dt = 0.8,V = V,E = E, uo = 'random')
u_res_3 = c2.mbo_zero_means()

#computing err
err1 = data.classification_error(u_res_1)
err2 = data.classification_error(u_res_2)
err3 = data.classification_error(u_res_3)
title1 = str('gl_supervised, %.2f error rate'%err1)
title2 = str('gl_zero_means, %.2f error rate'%err2)
title3 = str('mbo_zero_means, %.2f error rate'%err3)

print "plotting data"
data.plot_data(mode = 'Func', u = u_res_1,savefilename = 'gl_supervised.png',titlename = title1)
data.plot_data(mode = 'Func', u = u_res_2,savefilename = 'gl_zero_means_.png',titlename = title2)
data.plot_data(mode = 'Func', u = u_res_3,savefilename = 'mbo_zero_means.png',titlename = title3)








