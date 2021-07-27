#try stuff here
import matplotlib.pyplot as plt
import model_equation as meq
import plot as pl
import numpy as np
from scipy.signal import argrelextrema
import seaborn as sns
import pandas as pd
import cProfile

n="1"#"final"
filename="AC-DC_12"


par0 = {
    'K_ARAX':-3.5,#0.01,
    'n_ARAX':2,
    'K_XY':-2,
    'n_XY':2,
    'K_XZ':-1.25,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':0,
    'delta_X':1,
    
    'K_ARAY':-3.5,
    'n_ARAY':2,
    'K_YZ':-2,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':0,
    'delta_Y':1,
    
    'K_ZX':-2, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

}


par1 = {
    'K_ARAX':0.0,#0.01,
    'n_ARAX':2,
    'K_XY':-2,#
    'n_XY':2,
    'K_XZ':5,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':1,
    'delta_X':1,
    
    'K_ARAY':0.0,
    'n_ARAY':2,
    'K_YZ':-2,#0.01,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':1,
    'delta_Y':1,
    
    'K_ZX':-2, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

}


par2 = {
    'K_ARAX':-3.0,#0.01,
    'n_ARAX':2,
    'K_XY':-2,
    'n_XY':2,
    'K_XZ':0.05,#4.05
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':0,
    'delta_X':1,
    
    'K_ARAY':-3.0,
    'n_ARAY':2,
    'K_YZ':2,#0.01,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':1,
    'delta_Y':1,
    
    'K_ZX':2, 
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


pList, pdf= pl.load(n,filename)

ARA=meq.ARA
'''
ARA=np.array([0.000e+00, 3.125e-06, 6.250e-06, 1.250e-05, 2.500e-05, 5.000e-05,
       1.000e-04, 2.000e-04, 2.000e-01])

ARA=np.array([0,4.8828125e-05,0.0001953125,0.00078125,0.003125,0.0125,0.05,0.2])


ARA=np.logspace(-4.5,-2.,8,base=10)
'''

#print(d)
print("AC-DC " , meq.distance(ARA,par0))
#print("oscill " ,meq.distance(ARA,par1))
#print("increase " ,meq.distance(ARA,par2))
#print("dead " ,meq.distance(ARA,par3))

print("0 " , meq.distance(ARA,pList[0]))
#print("25 " ,meq.distance(ARA,p25))
#print("50 " ,meq.distance(ARA,p50))
#print("75 " ,meq.distance(ARA,p75))
print("991 " ,meq.distance(ARA,pList[990]))
print("996 " ,meq.distance(ARA,pList[995]))

print("1000 " ,meq.distance(ARA,pList[999]))


def plot(ARA,p,name,nb):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    for i,par in enumerate(p):
        

        X,Y,Z = meq.model(ARA,par,120)
        df_X=pd.DataFrame(X,columns=ARA)
        df_Y=pd.DataFrame(Y,columns=ARA)
        df_Z=pd.DataFrame(Z,columns=ARA)
        plt.subplot(len(p),3,(1+i*3))
        sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=0.2)
        plt.subplot(len(p),3,(2+i*3))
        sns.heatmap(df_Y, cmap ='Blues', vmin=0, vmax=0.2)
        plt.subplot(len(p),3,(3+i*3))
        sns.heatmap(df_Z, cmap ='Greens', vmin=0, vmax=0.2)

   # plt.savefig('smc_'+name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    plt.show()
    #plt.close()
'''
if __name__ == '__main__':
    cProfile.run('plot()')
'''
#plot(ARA,[par0,par1,par2,par3],filename,n)
#plot(ARA,[p0,p25,p50,p75,p100],filename,n)
plot(ARA,[pList[990],pList[995],pList[999]],filename,n)

