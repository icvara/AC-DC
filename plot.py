#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import abc_smc
import statistics
import os
from collections import Counter
import sys
from scipy.signal import argrelextrema
from matplotlib.colors import LogNorm, Normalize


filename="ACDC_X2"
n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
#['1','2','3','4','5','6','7','8','9','10','11','12']#,'16','17','18','19']
#n=['12','13','14','15','16','17','18','final']
#,'5','6','7','8',,'19','20','21','22','23','24']
#n=['1']

sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
#sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+filename)
import model_equation as meq
  

parlist=meq.parlist

##par from abc smc
def load(number= n,filename=filename,parlist=parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
        
    path = filename+'/smc/pars_' + number + '.out'
    dist_path = filename+'/smc/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = namelist)
    df['dist']=dist_output
    df=df.sort_values('dist',ascending=False)
    distlist= sorted(df['dist'])
    p=[]
    for dist in distlist:
        
        p_0=df[df['dist']==dist]
        p0=[]
        for n in namelist:
          p0.append(p_0[n].tolist()[0])
   
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
        sns.heatmap(df_X, cmap="Reds", norm=LogNorm())
        plt.subplot(len(p),3,(2+i*3))
        sns.heatmap(df_Y, cmap ='Blues', norm=LogNorm())
        plt.subplot(len(p),3,(3+i*3))
        sns.heatmap(df_Z, cmap ='Greens', norm=LogNorm())

    plt.savefig(name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.png', bbox_inches='tight')
   # plt.show()
    plt.close()


def par_plot(df,name,nb,parlist):

    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=2
    namelist=[]
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
            
      
    plt.savefig(name+"/plot/"+nb+'_Full_par_plot.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()


def bifurcation_plot(n,filename,p):
   # p,df= load(n,filename,parlist)
    ARA=np.logspace(-4.5,-2.,200,base=10)
    un,st,osc=calculateSS(ARA,p)
    M,m=getlimitcycle(ARA,osc,p,tt=500)
    for i,col in enumerate(['r','b','g']):
        plt.subplot(3,1,i+1)
        plt.plot(ARA,un[:,:,i],'--'+col)
        plt.plot(ARA,st[:,:,i],'-'+col)
        plt.plot(ARA,osc[:,:,i],'--'+col)
        plt.fill_between(ARA,M[:,0,i],m[:,0,i],alpha=0.2,facecolor=col)
        plt.fill_between(ARA,M[:,1,i],m[:,1,i],alpha=0.2,facecolor=col)
        plt.fill_between(ARA,M[:,2,i],m[:,2,i],alpha=0.2,facecolor=col)
        plt.yscale("log")
        plt.xscale("log")
    #plt.show()
    plt.savefig(filename+"/plot/"+n+'_Bifurcation.pdf', bbox_inches='tight')
    plt.close()

def getlimitcycle(ARA,ssl,par,tt=500):
    M=np.ones((len(ARA),3,3))*np.nan
    m=np.ones((len(ARA),3,3))*np.nan
    delta=10e-5
    transient=500
    for ai,a in enumerate(ARA):
            ss=ssl[ai]
            for si,s in enumerate(ss):
                if any(np.isnan(s)) == False:
                    init=[s[0]+delta,s[1]+delta,s[2]+delta]
                    X,Y,Z=meq.model([a],par,totaltime=tt,init=init)
                    M[ai,si,0]=max(X[transient:])
                    M[ai,si,1]=max(Y[transient:])
                    M[ai,si,2]=max(Z[transient:])
                    m[ai,si,0]=min(X[transient:])
                    m[ai,si,1]=min(Y[transient:])
                    m[ai,si,2]=min(Z[transient:])

                    max_list=argrelextrema(X[transient:], np.greater)
                    maxValues=X[transient:][max_list]
                    min_list=argrelextrema(X[transient:], np.less)
                    minValues=X[transient:][min_list]
                    if len(minValues)>3 and len(maxValues)>3:
                        maximaStability = abs(maxValues[-2]-minValues[-2])-(maxValues[-3]-minValues[-3])
                        if maximaStability > 0.01:
                            print("limit cycle not achieved for ARA["+str(ai)+"]:" + str(a) + " at st.s:"+ str(s))
                    else:
                        print("limit cycle not achieved for ARA["+str(ai)+"]:" + str(a) + " at st.s:"+ str(s))


    return M,m

def calculateSS(ARA,parUsed):
    #sort ss according to their stabilitz
    #create stability list of shape : arabinose x steady x X,Y,Z
    unstable=np.zeros((len(ARA),3,3))
    stable=np.zeros((len(ARA),3,3))
    oscillation=np.zeros((len(ARA),3,3))
    unstable[:]=np.nan
    stable[:]=np.nan
    oscillation[:]=np.nan

    for ai,a in enumerate(ARA):
        ss=meq.findss(a,parUsed)
        if len(ss) > 3:
            print("error: more than 3 steadystates")
        else:
            d = b = c=0 # can replace a,b,c by si, but allow to have osccilation on the same level
            for si,s in enumerate(ss):
                e=meq.stability(a,parUsed,[s])[0][0]
                if all(e<0):
                        stable[ai][d]=s
                        d+=1
                if any(e>0):
                    pos=e[e>0]
                    if len(pos)==2:
                        if pos[0]-pos[1] == 0:
                            oscillation[ai][b]=s
                            b+=1
                        else:
                            unstable[ai][c]=s
                            c+=1
                    else:
                        unstable[ai][c]=s 
                        c+=1                  
    return unstable,stable,oscillation

if __name__ == "__main__":
   
    if os.path.isdir(filename+'/plot') is False: ## if 'smc' folder does not exist:
        os.mkdir(filename+'/plot') ## create it, the output will go there
    
    ARA=meq.ARA
    for i in n:
      p, pdf= load(i,filename,parlist)
    
      plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,i)
      par_plot(pdf,filename,i,meq.parlist)

    bifurcation_plot('final',filename,p[0])

      

