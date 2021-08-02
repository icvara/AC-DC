
## equation for X only

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
import numpy as np

#par for abc_smc
initdist=40.
finaldist=2.5
priot_label=None
dtt=0.1
tt=120 #totaltime
tr=20 #transient time
node="X"

#list for ACDC
parlist = [ 
    #first node X param
    {'name' : 'K_ARAX', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    {'name' : 'n_ARAX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XY','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_XY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XZ','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_XZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta/alpha_X','lower_limit':0.0,'upper_limit':4.0},

    #Seconde node Y param
    {'name' : 'K_ARAY', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    {'name' : 'n_ARAY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_YZ','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_YZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta/alpha_Y','lower_limit':0.0,'upper_limit':4.0},

    #third node Z param
    {'name' : 'K_ZX','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_ZX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta/alpha_Z','lower_limit':0.0,'upper_limit':4.0},
]


ARA=np.logspace(-4.5,-2.,10,base=10) 


def Flow(X,Y,Z,ARA,par):
    #simplify version with less parameter    
    flow_x= 1 + (10**par['beta/alpha_X']-1)*(np.power(ARA,par['n_ARAX'])/( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))) 
    flow_x = flow_x / ( 1 + np.power((Z/10**(par['K_ZX']+par['beta/alpha_Z'])),par['n_ZX']))
    flow_x = flow_x - X

    flow_y = 1 + (10**par['beta/alpha_Y']-1)*( np.power(ARA,par['n_ARAY'])) / ( np.power(10**par['K_ARAY'],par['n_ARAY']) + np.power(ARA,par['n_ARAY']))
    flow_y = flow_y / ( 1 + np.power(X/10**(par['K_XY']+par['beta/alpha_X']),par['n_XY']))
    flow_y = flow_y - Y

    flow_z = 10**par['beta/alpha_Z']/( 1 + np.power(Y/10**(par['K_YZ']+par['beta/alpha_Y']),par['n_YZ']))
    flow_z = flow_z /( 1 + np.power(X/10**(par['K_XZ']+par['beta/alpha_X']),par['n_XZ']))
    flow_z = flow_z - Z

    return flow_x,flow_y,flow_z


def Integration(Xi,Yi,Zi, totaltime, dt, ch , pars ):
    X=Xi
    Y=Yi
    Z=Zi
    ti=0
    while (ti<totaltime):
        flow_x,flow_y,flow_z = Flow(Xi,Yi,Zi,ch,pars)
        Xi = Xi + flow_x*dt
        Yi = Yi + flow_y*dt
        Zi = Zi + flow_z*dt
        
        X=np.vstack((X,Xi))
        Y=np.vstack((Y,Yi))
        Z=np.vstack((Z,Zi))

        ti=ti+dt

    return X,Y,Z


def distance(x,pars,totaltime=tt, dt=dtt,trr=tr,N=node):

    X,Y,Z = model(x,pars,totaltime,dt)
    
    # transient time / dt
    transient = int(trr/dt)
 
    #range where oscillation is expected
    oscillation_ara=[2,7]
    
    A=X #default
    if N== "X":
      A=X
    if N== "Y":
      A=Y
    if N== "Z":
      A=Z    
    d_final=0
        
    for i in range(0,len(x)):
             # for local maxima
             max_list=argrelextrema(A[transient:,i], np.greater)
             maxValues=X[transient:,i][max_list]
             # for local minima
             min_list=argrelextrema(A[transient:,i], np.less)
             minValues=X[transient:,i][min_list]
     
     
             if i>oscillation_ara[0] and i<oscillation_ara[1]:           
                
                 if len(maxValues)>0 and len(maxValues)<2 and len(minValues)<2:
                     d= 1/len(maxValues) + 1
                
                 if len(maxValues)>=3 and len(minValues)>=3:  #if there is more than one peak
                    # print("max: " + str(len(maxValues)) + "   min:" + str(len(minValues)))
     
                     #here the distance is only calculated on the last two peaks
                     #d2=abs((maxValues[-1]-minValues[-1]) - (maxValues[-2]-minValues[-2]))/(maxValues[-2]-minValues[-2])  #maybe issue still here ? 
                     #d3=2*(minValues[-1])/(minValues[-1]+maxValues[-1]) #Amplitude of oscillation
                     
                     d2=abs((maxValues[-2]-minValues[-2]) - (maxValues[-3]-minValues[-3]))/(maxValues[-2]-minValues[-2])  #maybe issue still here ? 
                     d3=2*(minValues[-2])/(minValues[-2]+maxValues[-2]) #Amplitude of oscillation
                     d= d2+d3
     
                 else:
                     d=10 # excluded all the one without oscillation 
                     #d=abs(max(X[transient:,i])-max(X[transient:,(i+1)]))/max(X[transient:,i])
                     #this number can be tuned to help the algorythm to find good parameter....
                 #d=0 #v22 DC only
                 
             if i<oscillation_ara[0] or i>oscillation_ara[1]:  #notice than 2 inducer concentration are not precised here. no leave some place at transition dynamics
                 d1=  len(minValues)/(1+len(minValues)) # v14,21 with len(minValues)/(1+len(minValues)) #v15 10*len(minValues)/(1+len(minValues))
                 d2=  2*(max(A[transient:,i])-min(A[transient:,i]))/(max(A[transient:,i])+min(A[transient:,i]))
                 d= d1+d2
                 #d= 0 #v20 try to have repressilator
                
     
                 
             if i==oscillation_ara[0] or i==oscillation_ara[1]: 
                 d=0
            # print(d)
             d_final=d_final+d
        
    
   # d= 10*A[-1,0]/(A[-1,-1]+A[-1,0]) #try to valorise increase behaviour compare to dead one
   # print("diff   ", d)
    d_final=d_final+d
    
    if N=="ALL":
      dy=distance(x,pars,totaltime=tt, dt=dtt,trr=tr,N="Y")
      dz=distance(x,pars,totaltime=tt, dt=dtt,trr=tr,N="Z")
      d_final=d_final+dy+dz
        
    return d_final


def model(x,pars,totaltime=tt, dt=dtt):
    Xi=np.ones(len(x))*0.2
    Yi=np.zeros(len(x))
    Zi=np.zeros(len(x))
    X,Y,Z = Integration(Xi,Yi,Zi,totaltime,dt,x,pars)
    return X,Y,Z
