#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import abc_smc
import model_equation as meq
import statistics

from collections import Counter

#['1','2','3','4','5','6','7','8','9','10','11',
n=['1','2','3','4','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
filename="AC-DC_2"

##par from abc smc
def load(number= n,filename=filename):


    path = 'smc_'+filename+'/pars_' + number + '.out'
    dist_path = 'smc_'+filename+'/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = ['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ', 'beta_X','alpha_X','delta_X',
                                              'K_ARAY','n_ARAY','K_YZ','n_YZ', 'beta_Y','alpha_Y','delta_Y',
                                              'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z'])
    df['dist']=dist_output

    distlist= sorted(df['dist'])

    p_0=df[df['dist']==min(df['dist'])]
    p_25=df[df['dist']==distlist[250]]
    p_50=df[df['dist']==distlist[500]]
    p_75=df[df['dist']==distlist[750]]
    p_100=df[df['dist']==max(df['dist'])]
    #print(min(df['dist']),distlist[250],distlist[500],distlist[750],max(df['dist']))

    p_min=p_0
    p0=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
            p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
            p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
            p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
            p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]
    p_min=p_25
    p25=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
            p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
            p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
            p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
            p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]
    p_min=p_50
    p50=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
            p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
            p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
            p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
            p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]
    p_min=p_75
    p75=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
            p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
            p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
            p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
            p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]
    p_min=p_100
    p100=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
            p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
            p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
            p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
            p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]
    
    p0=abc_smc.pars_to_dict(p0)
    p25=abc_smc.pars_to_dict(p25)
    p50=abc_smc.pars_to_dict(p50)
    p75=abc_smc.pars_to_dict(p75)
    p100=abc_smc.pars_to_dict(p100)
    
    return p0,p25,p50,p75,p100, df


##plotting part

def plot(ARA,p,name,nb):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    for i,par in enumerate(p):
        

        X,Y,Z = meq.model(ARA,par)


        df_X=pd.DataFrame(X,columns=ARA)
        df_Y=pd.DataFrame(Y,columns=ARA)
        df_Z=pd.DataFrame(Z,columns=ARA)


        plt.subplot(len(p),3,(1+i*3))
        sns.heatmap(df_X, cmap="Reds", vmin=0, vmax=0.2)
        plt.subplot(len(p),3,(2+i*3))
        sns.heatmap(df_Y, cmap ='Blues', vmin=0, vmax=0.2)
        plt.subplot(len(p),3,(3+i*3))
        sns.heatmap(df_Z, cmap ='Greens', vmin=0, vmax=0.2)

    #plt.savefig("plot/"+name+'_'+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X[:,-1],Y[:,-1],Z[:,-1],'-')
    ax.view_init(30, 75)
    plt.savefig("plot/"+name+'_'+n+'_3D'+'.pdf', bbox_inches='tight')
    plt.close()


    plt.plot(X[:,-1],'-r')
    plt.plot(Y[:,-1],'-b')
    plt.plot(Z[:,-1],'-g')
    plt.ylim(0,1)
    plt.plot([20/0.1, 20/0.1], [0, 1], 'k--', lw=1)
    plt.savefig("plot/"+name+'_'+n+'_time'+'.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()
    '''



def par_plot(df,name,nb,parlist):
    listpar = ['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ', 'beta_X','alpha_X','delta_X',
                                              'K_ARAY','n_ARAY','K_YZ','n_YZ', 'beta_Y','alpha_Y','delta_Y',
                                              'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z']
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
        
    g=sns.pairplot(df[namelist], kind='kde', corner=True)

    for i,par in enumerate(namelist): 
       # g.axes[i,0].set_xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        g.axes[-1,i].set_xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        g.axes[i,0].set_ylim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))

    plt.savefig("plot/"+name+'_'+nb+'_Full_par_plot.pdf', bbox_inches='tight')
    plt.close()


    '''                                        
    

    
    
    sns.pairplot(df[['K_XY','n_XY','alpha_X','delta_X',
                      'K_YZ','n_YZ','alpha_Y','delta_Y',
                      'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z']], kind='kde')
    plt.savefig("plot/"+name+'_par_plot.pdf', bbox_inches='tight')
    
    plt.close()
    
    plt.savefig("plot/"+name+'_K_par_plot.pdf', bbox_inches='tight')
    
    plt.close()
  
    sns.pairplot(df[['beta_X','alpha_X','delta_X', 'beta_Y','alpha_Y','delta_Y', 'beta_Z','alpha_Z','delta_Z']], kind='kde')
    
    plt.savefig("plot/"+name+'_beta_par_plot.pdf', bbox_inches='tight')
    
    g = sns.pairplot(df[['K_ARAX','n_ARAX','K_XY']], kind='kde')

    
    '''
    #plt.show()

if __name__ == "__main__":
    ARA=meq.ARA
    for i in n:
      p0,p25,p50,p75,p100, pdf= load(i,filename)

      plot(ARA,[p0,p25,p50,p75,p100],filename,i)
     # par_plot(pdf,filename,i,meq.parlist)
