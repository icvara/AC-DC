## equation for Represilator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema



par = {
    'K_ARAX':5,#0.01,
    'n_ARAX':2,
    'K_ARAY':5,
    'n_ARAY':2,
    
    'K_ZX':0.01, 
    'K_XY':0.01,
    'K_YZ':0.01,
    
    'K_XZ':1e10,#4.05
    
    'n_ZX':2,
    'n_XZ':2,
    'n_XY':2,
    'n_YZ':2,

    'beta_Z':1,
    'beta_Y':1,
    'beta_X':1,

    'alpha_Z':0,
    'alpha_Y':1,
    'alpha_X':1,    

    'delta_X':1,
    'delta_Z':1,
    'delta_Y':1,
}


parlist = [ # list containing information of each parameter
    #first node X param
    {'name' : 'K_ARAX', 'lower_limit':4.5,'upper_limit':5.0}, #in log
    {'name' : 'n_ARAX','lower_limit':1.8,'upper_limit':2.0},
    {'name' : 'K_XY','lower_limit':0.01,'upper_limit':0.02},
    {'name' : 'n_XY','lower_limit':1.8,'upper_limit':2.0},
    {'name' : 'K_XZ','lower_limit':90.0,'upper_limit':100.0},
    {'name' : 'n_XZ','lower_limit':1.8,'upper_limit':2.0},
    {'name' : 'beta_X','lower_limit':1.0,'upper_limit':2.0},
    {'name' : 'alpha_X','lower_limit':0.0,'upper_limit':0.5},
    {'name' : 'delta_X','lower_limit':0.0,'upper_limit':1.0},


    #Seconde node Y param
    {'name' : 'K_ARAY', 'lower_limit':4.5,'upper_limit':5.0}, #in log
    {'name' : 'n_ARAY','lower_limit':1.0,'upper_limit':2.0},
    {'name' : 'K_YZ','lower_limit':0.01,'upper_limit':0.02},
    {'name' : 'n_YZ','lower_limit':1.8,'upper_limit':2.0},
    {'name' : 'beta_Y','lower_limit':1.0,'upper_limit':2.0},
    {'name' : 'alpha_Y','lower_limit':0.0,'upper_limit':0.5},
    {'name' : 'delta_Y','lower_limit':0.0,'upper_limit':1.0},


    #third node Z param
    {'name' : 'K_ZX','lower_limit':0.01,'upper_limit':0.02},
    {'name' : 'n_ZX','lower_limit':1.8,'upper_limit':2.0},
    {'name' : 'beta_Z','lower_limit':1.0,'upper_limit':2.0},
    {'name' : 'alpha_Z','lower_limit':0.0,'upper_limit':0.5},
    {'name' : 'delta_Z','lower_limit':0.0,'upper_limit':1.0},
]


#ARA =np.arange(0,0.2,0.005)
ARA=np.array([0.000e+00, 3.125e-06, 6.250e-06, 1.250e-05, 2.500e-05, 5.000e-05,
       1.000e-04, 2.000e-04, 2.000e-01])
ARA = np.array([0])

'''
def Distance(pars, y_data):
    model_X,model_Y,model_Z = Intergration(Xi,Yi,Zi,100,0.1,ARA,pars)
'''   

def Flow(X,Y,Z,ARA,par):
    flow_x= par['alpha_X'] +( np.power((par['beta_X']-par['alpha_X']) *ARA,par['n_ARAX'])) / ( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))
    flow_x= flow_x / ( 1 + np.power(Z/par['K_ZX'],par['n_ZX']))
    flow_x = flow_x - X*par['delta_X']

    flow_y= par['alpha_Y'] +( np.power((par['beta_Y']-par['alpha_Y']) *ARA,par['n_ARAY'])) / ( np.power(10**par['K_ARAY'],par['n_ARAY']) + np.power(ARA,par['n_ARAY']))
    flow_y= flow_y / ( 1 + np.power(X/par['K_XY'],par['n_XY']))
    flow_y = flow_y - Y*par['delta_Y']

    flow_z= par['alpha_Z'] + (par['beta_Z']-par['alpha_Z'])/( 1 + np.power(Y/par['K_YZ'],par['n_YZ']))
    flow_z= flow_z /( 1 + np.power(X/par['K_XZ'],par['n_XZ']))
    flow_z = flow_z - Z*par['delta_Z']

    return flow_x,flow_y,flow_z



def Integration(Xi,Yi,Zi, totaltime=100, dt=0.1, ch=ARA , pars=par ):
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

def distance(x,pars,totaltime=100, dt=0.1):
    
    X,Y,Z = model(x,pars,totaltime,dt)
    
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
        d2=abs((maxValues[-1] - maxValues[-2])/maxValues[-2])
        d3=2*(max(maxValues))/(min(minValues)+max(maxValues))
        d_final= d2+d3
 
    return d_final

def model(x,pars,totaltime=100, dt=0.1):
    Xi=np.ones(len(ARA))*0.5
    Yi=np.zeros(len(ARA))
    Zi=np.zeros(len(ARA))
    X,Y,Z = Integration(Xi,Yi,Zi,totaltime,dt,x,pars)
    return X,Y,Z


def plot(ARA,par):


    X,Y,Z = model(ARA,par)


    df_X=pd.DataFrame(X,columns=ARA)
    df_Y=pd.DataFrame(Y,columns=ARA)
    df_Z=pd.DataFrame(Z,columns=ARA)


    plt.subplot(1,3,1)
    sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=0.2)
    plt.subplot(1,3,2)
    sns.heatmap(df_Y, cmap ='Blues', vmin=0, vmax=0.2)
    plt.subplot(1,3,3)
    sns.heatmap(df_Z, cmap ='Greens', vmin=0, vmax=0.2)
    plt.savefig('heatmap'+'.pdf', bbox_inches='tight')
    #plt.show()
    plt.close()



    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X[:,0],Y[:,0],Z[:,0],'-')
    plt.savefig('1'+'.pdf', bbox_inches='tight')
    #plt.show()





