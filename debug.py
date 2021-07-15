#try stuff here
import matplotlib.pyplot as plt
import model_equation as meq
import plot
import numpy as np
from scipy.signal import argrelextrema
import seaborn as sns
import pandas as pd
import cProfile

n="final"#"final"
nv="Prange3"
filename=""


par0 = {
    'K_ARAX':-3.5,#0.01,
    'n_ARAX':2,
    'K_XY':0.01,
    'n_XY':2,
    'K_XZ':0.05,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':0,
    'delta_X':1,
    
    'K_ARAY':-3.5,
    'n_ARAY':2,
    'K_YZ':0.01,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':0,
    'delta_Y':1,
    
    'K_ZX':0.01, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

}


par1 = {
    'K_ARAX':0.0,#0.01,
    'n_ARAX':2,
    'K_XY':0.01,#
    'n_XY':2,
    'K_XZ':5,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':1,
    'delta_X':1,
    
    'K_ARAY':0.0,
    'n_ARAY':2,
    'K_YZ':0.01,#0.01,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':1,
    'delta_Y':1,
    
    'K_ZX':0.01, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

}


par2 = {
    'K_ARAX':-3.0,#0.01,
    'n_ARAX':2,
    'K_XY':0.01,
    'n_XY':2,
    'K_XZ':0.05,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':0,
    'delta_X':1,
    
    'K_ARAY':-3.0,
    'n_ARAY':2,
    'K_YZ':1,#0.01,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':1,
    'delta_Y':1,
    
    'K_ZX':1.01, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

}

par3 = {
    'K_ARAX':-3.0,#0.01,
    'n_ARAX':2,
    'K_XY':0.01,
    'n_XY':2,
    'K_XZ':0.05,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':1,
    'delta_X':1,
    
    'K_ARAY':-3.0,
    'n_ARAY':2,
    'K_YZ':1,#0.01,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':0,
    'delta_Y':1,
    
    'K_ZX':1.01, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

}



ARA=meq.ARA
ARA=np.array([0.000e+00, 3.125e-06, 6.250e-06, 1.250e-05, 2.500e-05, 5.000e-05,
       1.000e-04, 2.000e-04, 2.000e-01])

ARA=np.array([0,4.8828125e-05,0.0001953125,0.00078125,0.003125,0.0125,0.05,0.2])


ARA=np.logspace(-4.5,-2.,8,base=10)


#print(d)
print("AC-DC " , meq.distance(ARA,par0))
print("oscill " ,meq.distance(ARA,par1))
print("increase " ,meq.distance(ARA,par2))
print("dead " ,meq.distance(ARA,par3))

'''
i=4
plt.plot(X[:,i])
plt.plot(Y[:,i])
plt.plot(Z[:,i])

plt.show()


df_X=pd.DataFrame(X,columns=ARA)
df_Y=pd.DataFrame(Y,columns=ARA)
df_Z=pd.DataFrame(Z,columns=ARA)
plt.subplot(1,3,1)
sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=1)
plt.subplot(1,3,2)
sns.heatmap(df_Y, cmap ='Blues', vmin=0, vmax=1)
plt.subplot(1,3,3)
sns.heatmap(df_Z, cmap ='Greens', vmin=0, vmax=1)
plt.show()
'''



def plot():

    X,Y,Z=meq.model(ARA,par0)
    df_X=pd.DataFrame(X,columns=ARA)
    plt.subplot(2,2,1)
    sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=1)
    X,Y,Z=meq.model(ARA,par1)
    df_X=pd.DataFrame(X,columns=ARA)
    plt.subplot(2,2,2)
    sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=1)
    X,Y,Z=meq.model(ARA,par2)
    df_X=pd.DataFrame(X,columns=ARA)
    plt.subplot(2,2,3)
    sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=1)
    X,Y,Z=meq.model(ARA,par3)
    df_X=pd.DataFrame(X,columns=ARA)
    plt.subplot(2,2,4)
    sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=1)

    plt.show()
'''
if __name__ == '__main__':
    cProfile.run('plot()')
'''
#plot()
