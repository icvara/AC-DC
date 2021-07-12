#try stuff here

import model_equation as meq
import plot
import numpy as np
from scipy.signal import argrelextrema


n="5"

ARA=meq.ARA
p, pdf= plot.load(n)
p2=meq.par



X,Y,Z=meq.model(ARA,p2)

transient = int(20/0.1) # transient time / dt
# for local maxima
max_list=argrelextrema(X[transient:,0], np.greater)
maxValues=X[transient:,0][max_list]
# for local minima
min_list=argrelextrema(X[transient:,0], np.less)
minValues=X[transient:,0][min_list]

if len(maxValues)>0:
    d_final= 1/len(maxValues) + 2
else:
    d_final = 10e10
if len(maxValues)>4:
    #all time point
    #d2=abs((maxValues[1:] - maxValues[0:-1])/maxValues[0:-1])
    #d3=2*(np.sum(minValues[1:])/(np.sum(minValues[1:])+np.sum(maxValues[1:])))
    #d_final= np.sum(d2)+d3
    #last time point
    d2=abs(((maxValues[-1]-minValues[-1]) - (maxValues[-2]-minValues[-2]))/(maxValues[-2]-minValues[-2]))
    d3=2*(min(minValues))/(min(minValues)+max(maxValues))
    d_final= d2+d3
