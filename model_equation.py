
## equation for Represilator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
import numpy as np



#list for ACDC
parlist = [ 
    #first node X param
    {'name' : 'K_ARAX', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    {'name' : 'n_ARAX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XY','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_XY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XZ','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_XZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_X','lower_limit':0.5,'upper_limit':1.0},
    {'name' : 'alpha_X','lower_limit':0.0,'upper_limit':1.0},
    {'name' : 'delta_X','lower_limit':0.99,'upper_limit':1.0},


    #Seconde node Y param
    {'name' : 'K_ARAY', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    {'name' : 'n_ARAY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_YZ','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_YZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_Y','lower_limit':0.5,'upper_limit':1.0},
    {'name' : 'alpha_Y','lower_limit':0.0,'upper_limit':1.0},
    {'name' : 'delta_Y','lower_limit':0.99,'upper_limit':1.0},


    #third node Z param
    {'name' : 'K_ZX','lower_limit':-4.0,'upper_limit':-1.0},
    {'name' : 'n_ZX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_Z','lower_limit':0.5,'upper_limit':1.0},
    {'name' : 'alpha_Z','lower_limit':0.0,'upper_limit':1.0},
    {'name' : 'delta_Z','lower_limit':0.99,'upper_limit':1.0},
]


ARA=np.logspace(-4.5,-2.,10,base=10) 




def Flow(X,Y,Z,ARA,par):
   # flow_x= 10**par['alpha_X'] +( np.power((10**par['beta_X']-10**par['alpha_X']) *ARA,par['n_ARAX'])) / ( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))
    maxrate_x = (par['beta_X']-par['alpha_X'])
    if maxrate_x < 0 :
       maxrate_x=0  
    flow_x = par['alpha_X'] + (np.power(maxrate_x *ARA,par['n_ARAX']))/( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))
    flow_x = flow_x / ( 1 + np.power((Z/10**par['K_ZX']),par['n_ZX']))
    flow_x = flow_x - X*par['delta_X']

    maxrate_y = (par['beta_Y']-par['alpha_Y'])
    if maxrate_y < 0 :
      maxrate_y=0  
    flow_y = par['alpha_Y'] +( np.power(maxrate_y *ARA,par['n_ARAY'])) / ( np.power(10**par['K_ARAY'],par['n_ARAY']) + np.power(ARA,par['n_ARAY']))
    flow_y = flow_y / ( 1 + np.power(X/10**par['K_XY'],par['n_XY']))
    flow_y = flow_y - Y*par['delta_Y']

    flow_z = par['alpha_Z'] + (par['beta_Z']-par['alpha_Z'])/( 1 + np.power(Y/10**par['K_YZ'],par['n_YZ']))
    flow_z = flow_z /( 1 + np.power(X/10**par['K_XZ'],par['n_XZ']))
    flow_z = flow_z - Z*par['delta_Z']

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


def distance(x,pars,totaltime=120, dt=0.1):

    X,Y,Z = model(x,pars,totaltime,dt)
    
    # transient time / dt
    transient = int(20/dt)
 
    #range where oscillation is expected
    oscillation_ara=[2,7]

    d_final=0

    for i in range(0,len(x)):
        # for local maxima
        max_list=argrelextrema(X[transient:,i], np.greater)
        maxValues=X[transient:,i][max_list]
        # for local minima
        min_list=argrelextrema(X[transient:,i], np.less)
        minValues=X[transient:,i][min_list]


        if i>oscillation_ara[0] and i<oscillation_ara[1]:           
            '''
            if len(maxValues)>0 and len(maxValues)<2:
                d= 1/len(maxValues) + 1
            '''
            if len(maxValues)>=2:  #if there is more than one peak
                #here the distance is only calculated on the last two peaks
                #d2=abs(((maxValues[-1]-minValues[-1]) - (maxValues[-2]-minValues[-2]))/(maxValues[-2]-minValues[-2])) #stability of oscillation
                d2=abs(((maxValues[-1]-minValues[-1]) - (maxValues[0]-minValues[0]))/(maxValues[-1]-minValues[-1])) #stability of oscillation first and last peaks
                d3=2*(minValues[-1])/(minValues[-1]+maxValues[-1]) #Amplitude of oscillation
                d= d2+d3

            else:
                d=3
                #d=abs(max(X[transient:,i])-max(X[transient:,(i+1)]))/max(X[transient:,i])
                #this number can be tuned to help the algorythm to find good parameter....
            
        if i<oscillation_ara[0] or i>oscillation_ara[1]:  #notice than 2 inducer concentration are not precised here. no leave some place at transition dynamics
            d1=  len(minValues)/(1+len(minValues))
            d2=  2*(max(X[transient:,i])-min(X[transient:,i]))/(max(X[transient:,i])+min(X[transient:,i]))
            d= d1+d2
           

            
        if i==oscillation_ara[0] or i==oscillation_ara[1]: 
            d=0
      
        d_final=d_final+d
        
    
   # d= 10*X[-1,0]/(X[-1,-1]+X[-1,0]) #try to valorise increase behaviour compare to dead one
   # print("diff   ", d)
    d_final=d_final+d
        
    return d_final


def model(x,pars,totaltime=100, dt=0.1):
    Xi=np.ones(len(x))*0.2
    Yi=np.zeros(len(x))
    Zi=np.zeros(len(x))
    X,Y,Z = Integration(Xi,Yi,Zi,totaltime,dt,x,pars)
    return X,Y,Z







