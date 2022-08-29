#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os
from collections import Counter
import sys
from scipy.signal import argrelextrema
from matplotlib.colors import LogNorm, Normalize
import multiprocessing
import time
from functools import partial
from scipy.stats import gaussian_kde
from matplotlib import colors
from Bifurcation import *



color=sns.color_palette("colorblind")
colorGREEN=color[2]
colorBLUE=color[0]
colorRED=color[3]
colorPurple=color[4]

'''
filename="ACDC_X21ind"

n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
#n=['7']

#sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+filename)
import model_equation as meq
parlist=meq.parlist
'''

######################################################################33
#########################################################################
###########################################################################

def load(number, filename,parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    
    path = filename+'/smc/pars_' + number + '.out'
    dist_path = filename+'/smc/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = namelist)
    df['dist']=dist_output
    idx=np.argsort(df['dist'])
    df=df.sort_values('dist',ascending=False)
    p=[]

    for i in idx:
        p0=df.loc[i].tolist()
        p.append(pars_to_dict(p0,parlist))


    '''
    for dist in distlist:
        
        p_0=df[df['dist']==dist]
        p0=[]
        for n in namelist:
          p0.append(p_0[n].tolist()[0])
        p0=pars_to_dict(p0,parlist)
        p.append(p0)
    '''
    
    return p, df 




def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars
##plotting part


def plot(ARA,p,name,nb,tt=120):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    for i,par in enumerate(p):
        

        X,Y,Z = meq.model(ARA,par,totaltime=tt)
        df_X=pd.DataFrame(X,columns=ARA)
        df_Y=pd.DataFrame(Y,columns=ARA)
        df_Z=pd.DataFrame(Z,columns=ARA)


        plt.subplot(len(p),3,(1+i*3))
        sns.heatmap(df_X, cmap="Reds", norm=LogNorm())
        plt.xticks([])
        plt.ylabel('time')
        plt.subplot(len(p),3,(2+i*3))
        sns.heatmap(df_Y, cmap ='Blues', norm=LogNorm())
        plt.xticks([])
        plt.yticks([])
        plt.subplot(len(p),3,(3+i*3))
        sns.heatmap(df_Z, cmap ='Greens', norm=LogNorm())
        plt.xticks([])
        plt.yticks([])



    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    plt.savefig(name+"/plot/heatmap/"+nb+'_heatmap'+'.png', bbox_inches='tight')
    #plt.show()
    plt.close()



def par_plot(df_par,parlist):
    fig, axs = plt.subplots(len(parlist),len(parlist))#, constrained_layout=True, figsize=(10/inch_cm,10/inch_cm))

    for x in np.arange(len(parlist)):
        for y in np.arange(len(parlist)):

            px=parlist[x]['name']
            py=parlist[y]['name']

            if x==y:
                #axs[x,y] = sns.distplot(df_par[px])
                axs[x,y].hist(df_par[px],bins=15, density=True)


                axs[x,y].set_xlim(parlist[x]['lower_limit'],parlist[x]['upper_limit'])
                #sns.kdeplot(ax=axs[x, y],x=df_par[px])

            elif y>x:
                txt=df_par[px][0]
                corr=np.corrcoef(df_par[px],df_par[py])[0, 1] #pearson correlation
                corr_r=np.round(corr,2)
                axs[x,y].imshow([[corr],[corr]], vmin=-1, vmax=1, cmap="bwr",aspect="auto")
                axs[x,y].text(0,0.5,corr_r,fontsize=8,ha='center', va='center')

            elif y<x:
                #z = gaussian_kde(df_par[px])(df_par[py])
                #idx = z.argsort()
                #xx, yy, z = df_par[px][idx], df_par[py][idx], z[idx]
                #axs[x,y].scatter(xx,yy,c=z, s=0.1, cmap='viridis')# vmin=mindist, vmax=maxdist)
                axs[x,y].hist2d(df_par[py],df_par[px],bins=15, norm = colors.LogNorm())

                axs[x,y].set_xlim(parlist[y]['lower_limit'],parlist[y]['upper_limit'])
                axs[x,y].set_ylim(parlist[x]['lower_limit'],parlist[x]['upper_limit'])


            
            niceaxis(axs,x,y,px,py,parlist,parlist,6)

    plt.subplots_adjust(wspace=None, hspace=None)

    return axs
    


def niceaxis(axs,x,y,px,py,kx,ky,size):
        #axs[x,y].set_ylim(-0.1,1.1)
        #if x==0:
        #    axs[x,y].set_title(py,fontsize=size)
        
        if y==0:
            axs[x,y].set_ylabel(px,fontsize=size,rotation=45,ha='right')
            axs[x,y].tick_params(axis='y', labelsize=size-2)

            if x!=len(kx)-1:
                    axs[x,y].set_xticks([])
                    axs[x,y].set_xticklabels([])
            #axs[x,y].set_yticks([])

        if x==len(kx)-1:
            axs[x,y].set_xlabel(py,fontsize=size,rotation=45)
            axs[x,y].tick_params(axis='x', labelsize=size-2)

            if y!=0:
                    axs[x,y].set_yticks([])
                    axs[x,y].set_yticklabels([])

        if y!=0 and x!=len(kx)-1:
            axs[x,y].set_xticks([])
            axs[x,y].set_yticks([])
            axs[x,y].set_xticklabels([])
            axs[x,y].set_yticklabels([])
        #if x==len(kx)-1:
        #   axs[x,y].set_xlabel('AHL')
        return axs


def heatmap_allroound(ARA,filename,parlist,n,meq):


    fig,axs=plt.subplots(3,len(n),constrained_layout=True, figsize=(len(n),3))
    for i,ni in enumerate(n):
        p, df= load(ni,filename,meq.parlist)
        for j,jj in enumerate(np.array([-1,500,0])):
            X,Y,Z = meq.model(ARA,p[jj],totaltime=120)
       # print(X.shape)

            axs[j,i].imshow(np.log10(X), aspect="auto", cmap="Reds")
           
            d=np.round(meq.distance(ARA,p[jj]),2)
            axs[j,i].set_title(('d = ' + str(d)), fontsize=6)
            axs[j,i].tick_params(axis='y', labelsize=6)
            axs[j,i].tick_params(axis='x', labelsize=6)
            axs[j,i].set_xlabel("S",fontsize=6)
            axs[j,i].set_ylabel("time",fontsize=6)




def barplot(df,parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    mean=np.array(df.mean().tolist()[:-1])
    sd=np.array(df.std().tolist()[:-1])
    median=np.array(df.median().tolist()[:-1])
    x_pos = np.arange(len(parlist))


   # mean=sd/mean
    namelist=np.array(namelist)
    #idx=np.argsort(np.abs(sd))
    #idx=np.argsort(np.abs(sd/mean))
    idx=np.argsort(x_pos)

    plt.bar(x_pos,mean[idx],color='black')
    plt.errorbar(x_pos,mean[idx],yerr=sd[idx], ecolor='k', linestyle='')


    #plt.plot(x_pos,median[:-1],'ro')
    plt.xticks(x_pos, namelist[idx], rotation=45, fontsize=8)
    plt.ylabel("parameter mean")# fontsize=8)



def bifuplot(ARA,filename,p,meq,dummy):

    ss=meq.findss2(ARA,p) 
    A=meq.jacobianMatrix2(ARA,ss,p)
    J=np.nan_to_num(A)

    eigvals, eigvecs =np.linalg.eig(J)
    stability_array=np.apply_along_axis(getStability, 2, eigvals)

    idx=np.where(stability_array==2)

    m=np.copy(stability_array)
    m[m!=1]=0
    stable_matrix = m*ss[:,:,0]
    stable_matrix[stable_matrix ==0]=np.nan

    m=np.copy(stability_array)
    m[m!=2]=0
    oscil_matrix = m/2*ss[:,:,0]
    oscil_matrix[oscil_matrix ==0]=np.nan
  
    low=[]
    high=[]
    idx = np.argwhere(stability_array==2)
    
    c=0
    for ai,a in enumerate(ARA):
        l=np.nan
        h=np.nan
        if ai == idx[c][0]:
            init=ss[idx[c][0],idx[c][1]]+1e-10
            l,h = limitcycle(ai,ss,ARA,init,p,dummy,meq)
            l=l[0]
            h=h[0]
            if c!=len(idx)-1:
                c+=1
        low.append(l)
        high.append(h)

    m=np.copy(stability_array)
    m[m!=3]=0
    unstable_matrix = m/3*ss[:,:,0]
    unstable_matrix[unstable_matrix ==0]=np.nan

    plt.plot(ARA,stable_matrix,'-', c=colorRED)
    plt.plot(ARA,unstable_matrix,'--', c=colorRED)
    plt.fill_between(ARA,low,high,alpha=0.5,facecolor=colorBLUE)
    plt.plot(ARA,oscil_matrix,'--', c=colorBLUE)
    plt.plot(ARA,low,'-',c=colorBLUE)
    plt.plot(ARA,high,'-',c=colorBLUE)

    '''
    plt.plot(ARA,M[:,i,0],'-b',linewidth=1)
    plt.plot(ARA,m[:,i,0],'-b',linewidth=1)
    plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
    '''
    plt.yscale('log')
    plt.xscale('log')


def bifuplot_grid(axs,i,j,ARA,filename,p,meq,dummy):

    ss=meq.findss2(ARA,p) 
    A=meq.jacobianMatrix2(ARA,ss,p)
    J=np.nan_to_num(A)
    eigvals, eigvecs =np.linalg.eig(J)
    stability_array=np.apply_along_axis(getStability, 2, eigvals)

    idx=np.where(stability_array==2)

    m=np.copy(stability_array)
    m[m!=1]=0
    stable_matrix = m*ss[:,:,0]
    stable_matrix[stable_matrix ==0]=np.nan

    m=np.copy(stability_array)
    m[m!=2]=0
    oscil_matrix = m/2*ss[:,:,0]
    oscil_matrix[oscil_matrix ==0]=np.nan
  
    low=[]
    high=[]
    idx = np.argwhere(stability_array==2)
    if len(idx)>0:
        c=0
        for ai,a in enumerate(ARA):
            l=np.nan
            h=np.nan

            if ai == idx[c][0]:
                init=ss[idx[c][0],idx[c][1]]+1e-10
                l,h = limitcycle(ai,ss,ARA,init,p,dummy,meq)
                l=l[0]
                h=h[0]
                if c!=len(idx)-1:
                    c+=1
            low.append(l)
            high.append(h)
    else:
        low=np.ones(len(ARA))*np.nan
        high=np.ones(len(ARA))*np.nan


    m=np.copy(stability_array)
    m[m!=3]=0
    unstable_matrix = m/3*ss[:,:,0]
    unstable_matrix[unstable_matrix ==0]=np.nan

    axs[i,j].plot(ARA,stable_matrix,'-', c=colorRED)
    axs[i,j].plot(ARA,unstable_matrix,'--', c=colorRED)
    axs[i,j].fill_between(ARA,low,high,alpha=0.5,facecolor=colorBLUE)
    axs[i,j].plot(ARA,oscil_matrix,'--', c=colorBLUE)
    axs[i,j].plot(ARA,low,'-',c=colorBLUE)
    axs[i,j].plot(ARA,high,'-',c=colorBLUE)

    '''
    plt.plot(ARA,M[:,i,0],'-b',linewidth=1)
    plt.plot(ARA,m[:,i,0],'-b',linewidth=1)
    plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
    '''
    axs[i,j].set_yscale('log')
    axs[i,j].set_xscale('log')


    return axs

##############################################################################################################3   

if __name__ == "__main__":
   
    if os.path.isdir(filename+'/plot') is False: ## if 'smc' folder does not exist:
        os.mkdir(filename+'/plot') ## create it, the output will go there
    
    ARA=meq.ARA
    ARA=np.logspace(-8,-2.,200,base=10)

    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    

    n=['final']
    p, pdf= load(n[0],filename,meq.parlist)
    df=pdf

    plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,"moretime",tt=500)

    '''
   # ARA=ARA[[0,4,5,9]]
    for i in n:
      p, pdf= load(i,filename,meq.parlist)
    
      plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,i)
      par_plot(pdf,filename,i,meq.parlist,namelist)

    '''
 

    
    