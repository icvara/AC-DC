#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import abc_smc
import model_equation as meq
import statistics

from collections import Counter

##par from abc smc
n="final"#"final"
nv="Prange3"
filename=""

def load(number= n):

    path = 'smc_'+filename+'/pars_' + number + '.out'
    dist_path = 'smc_'+filename+'/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = ['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ', 'beta_X','alpha_X','delta_X',
                                              'K_ARAY','n_ARAY','K_YZ','n_YZ', 'beta_Y','alpha_Y','delta_Y',
                                              'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z'])
    df['dist']=dist_output

    p_min=df[df['dist']==min(df['dist'])]

    p=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
    p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
    p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
    p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
    p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]

    p_final=abc_smc.pars_to_dict(p)
    return p_final, df

## predefined par
par = {
    'K_ARAX':-4, #5
    'n_ARAX':2,
    'K_ARAY':-4,#5,
    'n_ARAY':2,
    
    'K_ZX':0.01, 
    'K_XY':0.01,
    'K_YZ':0.01,
    
    'K_XZ':1e10,#0.05,#4.05, #1e10
    
    'n_ZX':2,
    'n_XZ':2,
    'n_XY':2,
    'n_YZ':2,

    'beta_Z':1,
    'beta_Y':1,
    'beta_X':1,

    'alpha_Z':0,
    'alpha_Y':0,
    'alpha_X':0,    

    'delta_X':1,
    'delta_Z':1,
    'delta_Y':1,
}

ARA=np.array([0.000e+00, 3.125e-06, 6.250e-06, 1.250e-05, 2.500e-05, 5.000e-05,
       1.000e-04, 2.000e-04, 2.000e-01])

##plotting part

def plot(ARA,par,name):


    X,Y,Z = meq.model(ARA,par)


    df_X=pd.DataFrame(X,columns=ARA)
    df_Y=pd.DataFrame(Y,columns=ARA)
    df_Z=pd.DataFrame(Z,columns=ARA)


    plt.subplot(1,3,1)
    sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=0.2)
    plt.subplot(1,3,2)
    sns.heatmap(df_Y, cmap ='Blues', vmin=0, vmax=0.2)
    plt.subplot(1,3,3)
    sns.heatmap(df_Z, cmap ='Greens', vmin=0, vmax=0.2)
    plt.savefig("plot/"+name+'_heatmap'+'.pdf', bbox_inches='tight')
    #plt.show()
    plt.close()



    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X[:,-1],Y[:,-1],Z[:,-1],'-')
    ax.view_init(30, 75)
    plt.savefig("plot/"+name+'_3D'+'.pdf', bbox_inches='tight')
    plt.close()


    plt.plot(X[:,-1],'-r')
    plt.plot(Y[:,-1],'-b')
    plt.plot(Z[:,-1],'-g')
    plt.ylim(0,1)
    plt.plot([20/0.1, 20/0.1], [0, 1], 'k--', lw=1)
    plt.savefig("plot/"+name+'_time'+'.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()



def par_plot(df,name):
    sns.pairplot(df[['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ', 'beta_X','alpha_X','delta_X',
                                              'K_ARAY','n_ARAY','K_YZ','n_YZ', 'beta_Y','alpha_Y','delta_Y',
                                              'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z']], kind='kde')
    plt.savefig("plot/"+name+'_Full_par_plot.pdf', bbox_inches='tight')
    
    sns.pairplot(df[['K_XY','n_XY','alpha_X','delta_X',
                      'K_YZ','n_YZ','alpha_Y','delta_Y',
                      'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z']], kind='kde')
    plt.savefig("plot/"+name+'_par_plot.pdf', bbox_inches='tight')
    '''
    plt.close()
    sns.pairplot(df[['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ',
                                              'K_ARAY','n_ARAY','K_YZ','n_YZ',
                                              'K_ZX','n_ZX']], kind='kde')
    plt.savefig("plot/"+name+'_K_par_plot.pdf', bbox_inches='tight')
    
    plt.close()
    sns.pairplot(df[['beta_X','alpha_X','delta_X', 'beta_Y','alpha_Y','delta_Y', 'beta_Z','alpha_Z','delta_Z']], kind='kde')
    plt.savefig("plot/"+name+'_beta_par_plot.pdf', bbox_inches='tight')
    '''

ARA=meq.ARA
p, pdf= load(n)

plot(ARA,p,nv)
par_plot(pdf,nv)
