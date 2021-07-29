#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import abc_smc
import model_equation as meq
import statistics
import os
from collections import Counter
import sys

filename="AC-DC_25"
n=['1','2','3','4','5','6','7','8','9','10','11']
n=['12','13','14','15','16','17','18','final']
#,'5','6','7','8',,'19','20','21','22','23','24']
#n=['1']

sys.path.insert(0, '/users/ibarbier/AC-DC/smc_'+filename)
#sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/smc_'+filename)
import par as pr
  



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
    df=df.sort_values('dist',ascending=False)
    distlist= sorted(df['dist'])
    p=[]
    for dist in distlist:
        
        p_0=df[df['dist']==dist]
   
        p_min=p_0
        p0=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
                p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
                p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
                p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
                p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]
   
        p0=abc_smc.pars_to_dict(p0)
        p.append(p0)

    
    return p, df


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

    plt.savefig('smc_'+name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
   # plt.show()
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

    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=2
    namelist=[]
    maxdist=20
    mindist=4
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
 
    
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
            if i == j :
                plt.hist(df[par1])
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c=df['dist'], s=0.001, cmap='viridis')# vmin=mindist, vmax=maxdist)
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
                plt.ylim((parlist[j]['lower_limit'],parlist[j]['upper_limit']))
            if i > 0 and j < len(namelist)-1 :
                plt.xticks([])
                plt.yticks([])
            else:
                if i==0 and j!=len(namelist)-1:
                    plt.xticks([])
                    plt.ylabel(par2,fontsize=fonts)
                    plt.yticks(fontsize=fonts,rotation=90)
                if j==len(namelist)-1 and i != 0:
                    plt.yticks([])
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                else:
                    plt.ylabel(par2,fontsize=fonts)
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                    plt.yticks(fontsize=4,rotation=90)
            
      
    plt.savefig('smc_'+name+"/plot/"+nb+'_Full_par_plot.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()



if __name__ == "__main__":
    
    if os.path.isdir('smc_'+filename+'/plot') is False: ## if 'smc' folder does not exist:
        os.mkdir('smc_'+filename+'/plot') ## create it, the output will go there
    
    ARA=pr.ARA
    for i in n:
      p, pdf= load(i,filename)
    

      plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,i)
      par_plot(pdf,filename,i,pr.parlist)
      
