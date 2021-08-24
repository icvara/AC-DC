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


filename="ACDC_X2"
n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
n=['final']

sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
#sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+filename)
import model_equation as meq
  
parlist=meq.parlist


######################################################################33
#########################################################################
###########################################################################

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
   
        p0=pars_to_dict(p0)
        p.append(p0)

    
    return p, df

def pars_to_dict(pars):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars
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

   # plt.savefig(name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.png', bbox_inches='tight')
    plt.show()
    plt.close()


def plotALLX(ARA,p,name,nb):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    sizex=np.sqrt(len(p))
    sizey=np.sqrt(len(p))
    for i,par in enumerate(p):        
        X,Y,Z = meq.model(ARA,par)
        df_X=pd.DataFrame(X,columns=ARA)
        #df_Y=pd.DataFrame(Y,columns=ARA)
        #df_Z=pd.DataFrame(Z,columns=ARA)
        plt.subplot(sizex,sizey,i+1)
        sns.heatmap(df_X, cmap="Reds", norm=LogNorm())

   # plt.savefig(name+"/plot/"+nb+'ALLALL_heatmap'+'.pdf', bbox_inches='tight')
    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.png', bbox_inches='tight')
    plt.show()
    plt.close()


def par_plot(df,name,nb,parlist,namelist):
    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=2
 
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
    plt.savefig(name+"/plot/"+nb+'_par_plot.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()
    
def splitted_parplot(n,filename,parlist):
    namelist=[]
    for i,par in enumerate(parlist):
       namelist.append(parlist[i]['name'])
    namelist=np.array(namelist) 
    parlist=np.array(parlist)   
    p, pdf= load(n,filename,parlist)

    namelist2=namelist[[2,4,9,12]] #only K par
    parlist2=parlist[[2,4,9,12]]
    namelist3=namelist[[0,1,6,7,8,11]] #only B and activation
    parlist3=parlist[[0,1,6,7,8,11]]
    namelist4=namelist[[7,8,9,10,11]] #only Y par
    parlist4=parlist[[7,8,9,10,11]]
    par_plot(pdf,filename,(str(n)+'ALL'),parlist,namelist)
    par_plot(pdf,filename,(str(n)+'K'),parlist2,namelist2)
    par_plot(pdf,filename,(str(n)+'B'),parlist3,namelist3)
    par_plot(pdf,filename,(str(n)+'Y'),parlist4,namelist4)
        
def plot_alltime(n,filename,parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    parl = np.append(namelist,'dist')
    index=1
    size=round(np.sqrt(len(parl)))
    for i,name in enumerate(parl):
        plt.subplot(size,size,index)
        plt.tight_layout()
        for ni,nmbr in enumerate(n):
            p,df= load(nmbr,filename,parlist)
            sns.kdeplot(df[name],bw_adjust=.8,label=nmbr)
        #plt.ylim(0,1)
        if i < (len(parl)-2):
            plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        if index==size:       
          plt.legend(bbox_to_anchor=(1.05, 1))
        index=index+1
    plt.savefig(filename+"/plot/"+'ALLround_plot.pdf', bbox_inches='tight')
    plt.close()


def par_plot2(df,df2,name,nb,parlist,namelist):
    a = df2[df2['up']>0]
    a=a[a['down']==0]
    b = df2[df2['down']>0 ]
    b=b[b['up']==0]
    c = df2[df2['idk']>0]
    d = df2[df2['down']>0 ]
    d=d[d['up']>0]
    #plot the parameter with the slected parameter on top of it
    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=2
    
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
            if i == j :
                sns.kdeplot(df[par1],color='black',bw_adjust=.8,linewidth=0.5)
                #sns.kdeplot(c[par1],color='gray',bw_adjust=.8,linewidth=0.5)
                #sns.kdeplot(a[par1],color='green', bw_adjust=.8,linewidth=0.5)
                sns.kdeplot(b[par1],color='red',bw_adjust=.8,linewidth=0.5)
                #sns.kdeplot(d[par1],color='orange',bw_adjust=.8,linewidth=0.5)
                plt.ylabel("")
                plt.xlabel("")
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c='black', s=0.0001)# vmin=mindist, vmax=maxdist)
               # plt.scatter(c[par1],c[par2], color='black', s=0.0001)
               # plt.scatter(a[par1],a[par2], color='green', s=0.0001)
                plt.scatter(b[par1],b[par2], color='red', s=0.0001)                
                #plt.scatter(d[par1],d[par2], color='orange', s=0.0001)
                #plt.scatter(df2[par1],df2[par2], c='blue', s=0.001)
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
    plt.savefig(name+"/"+nb+'_selected_par_plot.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()


def plotselectedparoverall(n,filename,parlist):
     selected_index = np.loadtxt(filename+'/ACDC_par_index.out')
     criteria = np.loadtxt(filename +'/criteria.out')
     selected_index =[int(x) for x in selected_index]      
     ARA=np.logspace(-4.5,-2.,20,base=10)
     p, pdf= load(n,filename,parlist)
     pdf2=pdf.iloc[selected_index]
     p_selected =  np.take(p,selected_index) 
     pdf2['up']=criteria[:,0]
     pdf2['down']=criteria[:,1]
     pdf2['idk']=criteria[:,2]
         
     namelist=[]
     for i,par in enumerate(meq.parlist):
       namelist.append(parlist[i]['name'])
     namelist=np.array(namelist)

     namelist2=namelist[[2,4,9,12]] #only K par
     parlist2=parlist[[2,4,9,12]] 
     namelist3=namelist[[0,1,6,7,8,11]] #only B and activation
     parlist3=parlist[[0,1,6,7,8,11]]
     namelist4=namelist[[7,8,9,10,11]] #only Y par
     parlist4=parlist[[7,8,9,10,11]]
     
     par_plot2(pdf,pdf2,filename,n,parlist,namelist)
     par_plot2(pdf,pdf2,filename,'K',parlist2,namelist2)
     par_plot2(pdf,pdf2,filename,'B',parlist3,namelist3)
     par_plot2(pdf,pdf2,filename,'Y',parlist4,namelist4)
     


def bifurcation_plot(n,filename,pars):
   # p,df= load(n,filename,parlist)
    ARA=np.logspace(-4.5,-2.,200,base=10)
    #ARA=np.logspace(-4.5,-2.,10,base=10)

    for pi,p in enumerate(pars):
        un,st,osc=calculateSS(ARA,p)
        M,m=getlimitcycle(ARA,osc,p)
        for i,col in enumerate(['r','b','g']):
            plt.subplot(len(pars),3,i+1+3*pi)
            plt.plot(ARA,un[:,:,i],'--'+col)
            plt.plot(ARA,st[:,:,i],'-'+col)
            plt.plot(ARA,osc[:,:,i],'--', c='black')

            plt.plot(ARA,M[:,0,i],'-'+col,linewidth=1)
            plt.plot(ARA,M[:,1,i],'-'+col,linewidth=1)
            plt.plot(ARA,M[:,2,i],'-'+col,linewidth=1)
            plt.plot(ARA,m[:,0,i],'-'+col,linewidth=1)
            plt.plot(ARA,m[:,1,i],'-'+col,linewidth=1)
            plt.plot(ARA,m[:,2,i],'-'+col,linewidth=1)
            plt.fill_between(ARA,M[:,0,i],m[:,0,i],alpha=0.2,facecolor=col)
            plt.fill_between(ARA,M[:,1,i],m[:,1,i],alpha=0.2,facecolor=col)
            plt.fill_between(ARA,M[:,2,i],m[:,2,i],alpha=0.2,facecolor=col)
            plt.yscale("log")
            plt.xscale("log")
    plt.show()
    #plt.savefig(filename+"/plot/"+n+'_Bifurcation.pdf', bbox_inches='tight')
    plt.close()

def bifurcation_Xplot(ARA,n,filename,pars,c):
    sizex=round(np.sqrt(len(pars))+0.5)
    sizey=round(np.sqrt(len(pars)))

    for pi,p in enumerate(pars):
        ss,eig,un,st,os,M,m=calculateALL(ARA,p)
        #M,m=getlimitcycle(ARA,osc,p)
        plt.subplot(sizex,sizey,pi+1)
        plt.tight_layout()
        for i in np.arange(0,un.shape[1]):
                plt.plot(ARA,un[:,i,0],'--',c='orange',linewidth=1)
                plt.plot(ARA,st[:,i,0],'-r',linewidth=1)
                plt.plot(ARA,os[:,i,0],'--b',linewidth=1)
                plt.plot(ARA,M[:,i,0],'-b',linewidth=1)
                plt.plot(ARA,m[:,i,0],'-b',linewidth=1)
                plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')

        plt.yscale("log")
        plt.xscale("log")
    
        '''
        if pi%(sizex+1) == 0:
            plt.xticks([])
        if pi%sizey == 0:
            plt.yticks([])
        else:
            plt.xticks([])
            plt.yticks([])
        '''
    plt.show()
    #plt.savefig(filename+'/'+str(c)+'_XBifurcation.pdf', bbox_inches='tight')
    plt.close()



##############################################3Bifurcation part 
################################################################################

def getminmax(X,Y,Z,transient):
    M=np.ones(3)*np.nan
    m=np.ones(3)*np.nan
    M[0]=max(X[transient:])
    M[1]=max(Y[transient:])
    M[2]=max(Z[transient:])
    m[0]=min(X[transient:])
    m[1]=min(Y[transient:])
    m[2]=min(Z[transient:])

    return M,m

def getpeaks(X,transient):
    max_list=argrelextrema(X[transient:], np.greater)
    maxValues=X[transient:][max_list]
    min_list=argrelextrema(X[transient:], np.less)
    minValues=X[transient:][min_list]

    return maxValues, minValues

def reachss(ssa,X,par,a):
    thr= 0.001
    out=False

    for ss in ssa:
        if np.all(np.isnan(ss)) == False:
            A=meq.jacobianMatrix(a,ss[0],ss[1],ss[2],par)
            eigvals, eigvecs =np.linalg.eig(A)
            sse=eigvals.real
            if np.all(sse<0):
                if abs((X[-2]-ss[0])/X[-2]) < thr:
                         out= True

    return out


def limitcycle(ai,ss,ARA,init,par,dummy,X=[],Y=[],Z=[],transient=500,count=0):
    threshold=0.01
    tt=200
    c=count
    #init=[init[0] + 10e-5,init[1] + 10e-5,init[2] + 10e-5]
    ssa=ss[ai]
    x,y,z=meq.model([ARA[ai]],par,totaltime=tt,init=init)
    X=np.append(X,x)
    Y=np.append(Y,y)
    Z=np.append(Z,z)

    M = m = np.nan

    maxValues, minValues = getpeaks(X,transient)

    if len(minValues)>4 and len(maxValues)>4:
        maximaStability = abs((maxValues[-2]-minValues[-2])-(maxValues[-3]-minValues[-3]))/(maxValues[-3]-minValues[-3]) #didn't take -1 and -2 because, i feel like -1 is buggy sometimes...
        if maximaStability > threshold:
            if reachss(ssa,X,par,ARA[ai])==False:
                #if we didn't reach the stability repeat the same for another 100 time until we reach it
                initt=[X[-2],Y[-2],Z[-2]] #take the -2 instead of -1 because sometimes the -1 is 0 because of some badly scripted part somewhere
                c=c+1
                if c<10:
               # if reachsteadystate(a,initt,par) == False:
                    M,m = limitcycle(ai,ss,ARA,initt,par,dummy,X,Y,Z,count=c)            
                if c==10:
                        #here the issue comes probably from 1) strange peak 2)very long oscillation
                        #here I try to get rid of strange peak , with multiple maxima and minima by peak. for this I take the local maximun and min of each..
                        #the issue here is to create artefact bc in the condition doesn't specify this kind of behaviour

                        maxValues2 = getpeaks(maxValues,0)[0]  
                        minValues2 = getpeaks(minValues,0)[1]
                        maximaStability2 = abs((maxValues2[-2]-minValues2[-2])-(maxValues2[-3]-minValues2[-3]))/(maxValues2[-3]-minValues2[-3]) #didn't take -1 and -2 because, i feel like -1 is buggy sometimes...
                        if maximaStability2 < threshold:
                            M,m = getminmax(X,Y,Z,transient=transient)
                        else:
                            #very long oscillation?
                            print("no limit cycle: probably encounter stable point at {} arabinose at p{}".format(ARA[ai],dummy))
                            plt.plot(X[transient:])
                            plt.yscale("log")
                            plt.show() 

        else:

            M,m = getminmax(X,Y,Z,transient=transient)
         
    else:
       # print("no enough oscillation: " + str(len(minValues)))
        if reachss(ssa,X,par,ARA[ai])==False:
            #print("homoclinic")
            initt=[X[-2],Y[-2],Z[-2]]
            c=c+1
            if c<10:
                M,m = limitcycle(ai,ss,ARA,initt,par,dummy,X,Y,Z,count=c)  
            if c==10: 
                #very long oscillation?          
                print("error in limit cycle ara = {}, p{}".format(ARA[ai],dummy))
                plt.plot(X[transient:])
                plt.yscale("log")
                plt.show()


    return M,m

def getEigen(ARA,par,s):
    A=meq.jacobianMatrix(ARA,s[0],s[1],s[2],par)
    eigvals, eigvecs =np.linalg.eig(A)
    sse=eigvals.real
    return sse #, np.trace(A), np.linalg.det(A)

def getpar(i,df):
    return pars_to_dict(df.iloc[i].tolist())

def calculateALL2(ARA,parUsed, dummy):
    #sort ss according to their stabilitz
    #create stability list of shape : arabinose x steady x X,Y,Z 
    nNode=3 # number of nodes : X,Y,Z
    nStstate= 5 # number of steady state accepted by. to create the storage array
  #  ss=np.ones((len(ARA),nStstate,nNode))*np.nan 
    eig= np.ones((len(ARA),nStstate,nNode))*np.nan 
    unstable=np.ones((len(ARA),nStstate,nNode))*np.nan
    stable=np.ones((len(ARA),nStstate,nNode))*np.nan
    oscillation=np.ones((len(ARA),nStstate,nNode))*np.nan
    homoclincic=np.ones((len(ARA),nStstate,nNode))*np.nan
    M=np.ones((len(ARA),nStstate,nNode))*np.nan
    m=np.ones((len(ARA),nStstate,nNode))*np.nan


    delta=10e-10 #perturbation from ss
    ss=meq.findss2(ARA,parUsed) 
    A=meq.jacobianMatrix2(ARA,ss,parUsed)

    for i in np.arange(0,len(ARA)):
        for j in np.arange(0,ss.shape[1]):
            if np.any(np.isnan(A[i][j]))==False:
                eigvals, eigvecs =np.linalg.eig(A[i][j])
                eig[i,j]=eigvals.real

                if any(eig[i,j]>0):
                    pos=eig[i,j][eig[i,j]>0]
                    if len(pos)==2:
                            if pos[0]-pos[1] == 0:                                
                                init=[ss[i,j,0]+delta,ss[i,j,1]+delta,ss[i,j,2]+delta]
                                M[i,j],m[i,j] = limitcycle(i,ss,ARA,init,parUsed,dummy)###
                                if np.isnan(M[i,j][0]):
                                    homoclincic[i][j]=ss[i,j] 

                                else:
                                    oscillation[i][j]=ss[i,j]
                            else:
                                unstable[i,j]=ss[i,j]
                    else:
                        unstable[i,j]=ss[i,j]
                else:
                    if np.all(eig[i,j]<0):
                        stable[i,j]=ss[i,j]
                    else:
                       unstable[i,j]=ss[i,j]
    return ss,eig,unstable,stable,oscillation,homoclincic,M,m


def findbifurcation(ARA,st,un,os,hc, dummy=0):
    bifu=np.zeros((len(ARA),un.shape[1])) 
    transition=np.zeros((len(ARA))) 
    os2=binarize(os)
    un2=binarize(un)
    st2=binarize(st)
    hc2=binarize(hc)

    #saddle node (1) : when two stable states (st,st) cohexist , transition type 1 st->st
    multistable=np.sum(st2,axis=1)
    multistable[multistable<2]=0
    v1=un2+multistable[:,None]
    v1[v1<3]=0
    v1=st2 +multistable[:,None] + os2
    saddlei_neg=np.where((v1[:-1]-v1[1:])==-3)
    saddlei_pos=np.where((v1[:-1]-v1[1:])==3)
    bifu[saddlei_neg[0]+1,saddlei_neg[1]]=1
    bifu[saddlei_pos[0],saddlei_pos[1]]=1
    transition[saddlei_neg[0]+1] = 1
    transition[saddlei_pos[0]] = 1

    #saddle node (1) : when two stable states (st,os) cohexist , transition type 2 st->os
    v2=np.sum(st2,axis=1)
    v2[v2>1]=1
    v1=os2+v2[:,None]
    v1[v1<2]=0
    saddlei_neg=np.where((v1[:-1]-v1[1:])==-2)
    saddlei_pos=np.where((v1[:-1]-v1[1:])==2)
    bifu[saddlei_neg[0]+1,saddlei_neg[1]]=1
    bifu[saddlei_pos]=1
    transition[saddlei_neg[0]+1] = 2
    transition[saddlei_pos[0]] = 2

    #homclinic (4) : when hc and st cohexist, transition type 4 os->st 
    v1=os2*10+hc2*4+st2
    hci_neg=np.where((v1[:-1]-v1[1:])==-6)
    hci_pos=np.where((v1[:-1]-v1[1:])==6)
    bifu[hci_neg[0]+1,hci_neg[1]]=4
    bifu[hci_pos]=4
    transition[hci_neg[0]+1] = 4
    transition[hci_pos[0]] = 4

    #when no other on same line mean that we have probably homoclinic..
    hci_neg=np.where((v1[:-1]-v1[1:])==-10)
    hci_pos=np.where((v1[:-1]-v1[1:])==10)
    if len(hci_neg[0])>0:
        print("please, check homoclinic for p" + str(dummy))
        bifu[hci_neg[0]+1,hci_neg[1]]=4
        bifu[hci_pos]=4
        transition[hci_neg[0]+1] = 4
        transition[hci_pos[0]] = 4

    #hopf (3) : when hc and st cohexist, transition type 3 st->os withotu hysteresis 
    hfi_neg=np.where((v1[:-1]-v1[1:])==-9)
    hfi_pos=np.where((v1[:-1]-v1[1:])==9)
    bifu[hfi_neg[0]+1,hfi_neg[1]]=3
    bifu[hfi_pos[0],hfi_pos[1]]=3
    transition[hfi_neg[0]+1] = 3
    transition[hfi_pos[0]] = 3

    #print(bifu)
    return bifu, transition

def countbifurcation(bifu):
    saddle = len(bifu[bifu==1])
    Hopf = len(bifu[bifu==3])
    homoclinic = len(bifu[bifu==4])
    return [saddle,Hopf,homoclinic]

    
def getmaxmultistability(os,st):
    v1=os
    v2=v1[:,:,0]/v1[:,:,0]
    v2[np.isnan(v2)]=0
    w1=st
    w2=w1[:,:,0]/w1[:,:,0]
    w2[np.isnan(w2)]=0

    total=np.sum((v2+w2),axis=1)
    return max(total)

def binarize(v1):
    v2=v1[:,:,0]/v1[:,:,0]
    v2[np.isnan(v2)]=0
    return v2


####################################################
###################################################PARALLELISATION HERE
######################################################

def isoscillationandstability(ARA,M,st,threshold=0.2):
    #check if we have limit cycle and stable state at the same ARA concentration 
    #isolate the behaviour at high ara (up) and low ara (down)
    nb = round(len(ARA)*threshold) # number of concentration required to be counted
    up=0
    down=0
    idk=0
    bdown=[]
    bup=[]
    t1=[]
    t2=[] 
    ACDC=False
    
    a=np.unique(np.argwhere(M>0)[:,0])
    b=np.unique(np.argwhere(st>0)[:,0]) 
    x=b[1:-1]-b[0:-2] #when the output give >1, it means there is a gap between the stable ss
    ind=np.where(x>1)   
    if len(ind)>1:
      print("error in attribution: more than 2 stable state")      
    if len(ind[0])==0: #only one steady state, this is a bit unlikely
      #let's put them in in special category atm
      idk=len(b)                
    else:
      index=ind[0][0]
      bdown=b[:(index+1)]
      bup=b[(index+1):]          
    for c1 in a:
      for c2 in bdown:
            t1.append(c1==c2)
      for c3 in bup:
            t2.append(c1==c3)
    down=sum(t1)
    up=sum(t2)   
    if up>nb or down>nb or idk>nb:
      ACDC=True   
    return ACDC,up,down, idk


def selectACDCpar(ARA,p,iter):
    ss,eig,un,st,osc,M,m=calculateALL(ARA,p)
    ACDC,up,down,idk=isoscillationandstability(ARA,M,st)
    if ACDC:
      return p,up,down,idk

def selectALL_ACDCpar(ARA,p,ncpus=40):
    pool = multiprocessing.Pool(ncpus)
    Npars=len(p)
    results = pool.map(func=partial(selectACDCpar, ARA, p),iterable=range(Npars), chunksize=10)
    pool.close()
    pool.join()    
    end_time = time.time()
    print(f'>>>> Loop processing time: {end_time-start_time:.3f} sec on {ncpus} CPU cores.')
    accepted_par = [result[0] for result in results]
    criteria = [result[1:3] for result in results]   
    return accepted_par, criteria

def run(n,filename,ncpus=40):
    ARA=np.logspace(-4.5,-2.,20,base=10)
    p, pdf= load(n,filename,meq.parlist)
    #accepted_par, criteria = selectALL_ACDCpar(ARA,p,ncpus)
    criteria=[]
    accepted_index=[]
    p=p[0:10]
    for i,pi in enumerate(p):
        ss,eig,un,st,osc,M,m=calculateALL(ARA,pi)
        ACDC,up,down,idk=isoscillationandstability(ARA,M,st)
        if ACDC:
            criteria.append([up,down,idk])
            accepted_index.append(i)
    np.savetxt(filename +'/criteria.out', criteria)
    np.savetxt(filename +'/ACDC_par_index.out', accepted_index)
    selected_index = np.loadtxt(filename+'/ACDC_par_index.out')
    criteria = np.loadtxt(filename +'/criteria.out')
    selected_index =[int(x) for x in selected_index]
    p_selected =  np.take(p,selected_index)
    bifurcation_Xplot(ARA,'final',filename,p_selected,c='selected')


def runBifurcation(ARA,pars, filename,n,index):
    sizex = round(np.sqrt(len(pars)))
    sizey = round(np.sqrt(len(pars))+0.5)

    max_stability=[]
    count_bifurcation=[]
    bifurcation_transition=[]

    for pi,p in enumerate(pars):
        s,eig,un,st,os,hc,M,m=calculateALL2(ARA,p,dummy=pi+index) 
        bifu,trans=findbifurcation(ARA,st,un,os,hc,dummy=pi+index)
        cbifu=countbifurcation(bifu)
        maxst=getmaxmultistability(os,st)
        max_stability.append(maxst)
        count_bifurcation.append(cbifu)
        bifurcation_transition.append(trans)


        plt.subplot(sizex,sizey,pi+1)
        plt.tight_layout()
        for i in np.arange(0,un.shape[1]):
            plt.plot(ARA,un[:,i,0],'--',c='orange',linewidth=.1)
            plt.plot(ARA,st[:,i,0],'-r',linewidth=.1)
            plt.plot(ARA,os[:,i,0],'--b',linewidth=.1)
            plt.plot(ARA,hc[:,i,0],'--g',linewidth=.1)
            plt.plot(ARA,M[:,i,0],'-b',linewidth=.1)
            plt.plot(ARA,m[:,i,0],'-b',linewidth=.1)
            plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
            plt.text(ARA[0],1,('p'+str(pi+index)),fontsize=1)
            plt.text(ARA[0],10,('S:{} Hf: {} Hmc: {}'.format(cbifu[0],cbifu[1],cbifu[2])),fontsize=1)
            plt.tick_params(axis='both', which='major', labelsize=2)

        plt.yscale("log")
        plt.xscale("log")

    np.savetxt(filename+"/"+str(index)+'_'+str(n)+'max_stability.out', max_stability)
    np.savetxt(filename+"/"+str(index)+'_'+str(n)+'count_bifurcation.out', count_bifurcation)
    np.savetxt(filename+"/"+str(index)+'_'+str(n)+'bifurcation_transition.out', bifurcation_transition)


    plt.savefig(filename+"/"+str(index)+'_'+str(n)+'XBifurcationplot.pdf', bbox_inches='tight')
    plt.show()


def runBifurcations(n,filename,ARAlen=20,ncpus=40):
    ARA=np.logspace(-4.5,-2.,ARAlen,base=10)
    p, pdf= load(n,filename,meq.parlist)
    for i in [0,1,2,3]:
        psubset=p[100*i:99+100*i]
        runBifurcation(ARA,psubset,filename,n,100*i)



################################BUILDING AREA
#run('final',filename,ncpus=40)
'''

selected_index = np.loadtxt(filename+'/ACDC_par_index.out')
criteria = np.loadtxt(filename +'/criteria.out')
selected_index =[int(x) for x in selected_index]
p_selected =  np.take(p,selected_index)
bifurcation_Xplot(ARA,'final',filename,p_selected,c='selected')
'''


    

runBifurcations('final',filename,ARAlen=50)




##############################################################################################################3   
'''
if __name__ == "__main__":
   
    if os.path.isdir(filename+'/plot') is False: ## if 'smc' folder does not exist:
        os.mkdir(filename+'/plot') ## create it, the output will go there
    
    ARA=meq.ARA
    ARA=np.logspace(-4.5,-2.,20,base=10)
   # plot_alltime(n,filename,meq.parlist)
    
    #for i in n:
    #  p, pdf= load(i,filename,meq.parlist)
    
     # plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,i)
     # par_plot(pdf,filename,i,meq.parlist)

    #bifurcation_plot('final',filename,p[1])
    p, pdf= load('final',filename,meq.parlist)
    index=[]
    for j in np.arange(0,9):
      acdc_p=[]     
      bifurcation_Xplot(ARA,'final',filename,p[(0+100*j):(99+100*j)],c=j)
      for i,pi in enumerate(p[(0+100*j):(99+100*j)]):
          ss,eig,un,st,osc,M,m=calculateALL(ARA,pi)
          if isoscillationandstability(ARA,M,st):
               acdc_p.append(pi)
               index.append(int(i+j*100)) 
          np.savetxt(str(j)+'_'+filename +'_selectedpar.out',index)
 '''     
        

    
    