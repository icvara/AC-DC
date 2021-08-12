#compare parameter between abc-smc
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
from matplotlib.colors import LogNorm, Normalize
from scipy.signal import argrelextrema


filename=["ACDC_X2","ACDC_Y2","ACDC_Z2"]#,"ACDC_all"]
#filename=['ACDC_X2']
#filename=['ACDC_X','ACDC_1ind']
n=['final']
#n=['1','2','3','4','5','6','7','8','9','10','11','12','final']#'13','14','15','final']
#n=['1','2','3','4','5','6','7','8','9','10','11','12','13','final']#,'12','13','14','final']#,'15']#,'final']

path='C:/Users/Administrator/Desktop/Modeling/AC-DC/'
path='/users/ibarbier/AC-DC/'
sys.path.insert(0, path + filename[0])

import model_equation as meq
  
parlist=meq.parlist
namelist=[]
for i,par in enumerate(parlist):
    namelist.append(parlist[i]['name'])


par0 = {
    'K_ARAX':-3.5,#0.01,
    'n_ARAX':2,
    'K_XY':-2.5,
    'n_XY':2,
    'K_XZ':-1.55,#-1.25
    'n_XZ':2,
    'beta_X':1,
    'alpha_X':0,
    'delta_X':1,
    
    'K_ARAY':-3.5,
    'n_ARAY':2,
    'K_YZ':-3.5,
    'n_YZ':2,
    'beta_Y':1,
    'alpha_Y':0,
    'delta_Y':1,
    
    'K_ZX':-2.5, 
    'n_ZX':2,
    'beta_Z':1,
    'alpha_Z':0,
    'delta_Z':1,

    'beta/alpha_X':2,
    'beta/alpha_Y':2,
    'beta/alpha_Z':2


}

def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars

def load(number= n,filename=filename,parlist=parlist):
    namelist=[]
    number=str(number)
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    filepath = path+filename+'/smc/pars_' + number + '.out'
    dist_path =  path+filename+'/smc/distances_' + number + '.out'
    raw_output= np.loadtxt(filepath)
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
        p0=pars_to_dict(p0,parlist)
        p.append(p0)    
    return p, df


def get_stats(filename,namelist):
    stats_df = pd.DataFrame( columns = ['par','file','mean','sd','mode'])
    parl = np.append(namelist,'dist')
   # for fi,fnm in enumerate(filename):
    fnm=filename[0]
    p,df= load(n[0],fnm,parlist)
    mean=np.mean(df).tolist()
    sd=np.std(df).tolist()
    mode=stats.mode(df)[0][0]
    new_row={'par':parl,'file':[fnm]*len(parl),'mean':mean,'sd':sd,'mode':mode}
    df2=pd.DataFrame(new_row)
    stats_df =stats_df.append(df2)
    return stats_df


def bar_plot(filename,namelist, t="mean"):
    stats_df=get_stats(filename,namelist)
    # set width of bars
    barWidth = 0.20
    # Set position of bar on X axis
    r1 = np.arange(len(parl))

    #mean
    if t=="mean":
        for i,nm in enumerate(filename):
            v=stats_df[stats_df['method']==nm]
            plt.bar((r1+barWidth*i),v['mean'],yerr=v['sd'], capsize=2,width=barWidth, label=nm)

        plt.xlabel('par', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(parl))], parl)
        plt.legend()
        plt.show()

    #mode       
    if t == "mode":
        for i,nm in enumerate(filename):
            v=stats_df[stats_df['method']==nm]
            plt.bar((r1+barWidth*i),v['mode'],width=barWidth, label=nm)

        plt.xlabel('par', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(parl))], parl)
        plt.legend()
        plt.show()


def plot_compare(n,filename,namelist):
    parl = np.append(namelist,'dist')
    index=1
    size=round(np.sqrt(len(parl)))
    for i,name in enumerate(parl):
        plt.subplot(size,size,index)
        plt.tight_layout()
        for fi,fnm in enumerate(filename):
            p,df= load(n,fnm,parlist)
            sns.kdeplot(df[name],bw_adjust=.8,label=fnm)
        #plt.ylim(0,1)
        if i < (len(parl)-2):
            plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        index=index+1
        if index==5:       
            plt.legend(bbox_to_anchor=(1.05, 1))

    #sns.kdeplot(df['K_XZ'])
    plt.savefig(str(filename)+str(n)+"_compareplot.pdf", bbox_inches='tight')
    plt.show()

#plot_compare(n[0],filename,namelist)


def plot_alltime(filename,namelist):
    parl = np.append(namelist,'dist')
    index=1
    for i,name in enumerate(parl):
        plt.subplot(4,4,index)
        plt.tight_layout()
        for ni,nmbr in enumerate(n):
            p,df= load(nmbr,filename[0],parlist)
            sns.kdeplot(df[name],bw_adjust=.8,label=nmbr)
        #plt.ylim(0,1)
        if i < (len(parl)-2):
            plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        index=index+1
        #if index==5:       
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

#plot_alltime(['ACDC_X2'],namelist)

def plotdistpar(filename,namelist):
    index=1
    for ni,nb in enumerate(n):
        p,df= load(nb,filename[0],parlist)
        for i,name in enumerate(namelist):
            plt.subplot(len(n),len(namelist),index)
           # plt.tight_layout()
            plt.scatter(df['dist'],df[name],s=1)
            mean=np.mean(df[name]).tolist()
            mode=stats.mode(df[name])[0][0]

            plt.plot([0,40],[mean,mean],'r',label="mean")
            plt.plot([0,40],[mode,mode],'g',label="meode")

            plt.ylim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            plt.ylabel(name)
            index=index+1
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()


'''
ARA=np.logspace(-4.5,-2.,10,base=10)
p,df= load(n[0],filename[0],parlist)

stdf=get_stats(filename,namelist)
pmean=pars_to_dict(stdf['mean'])
pmode=pars_to_dict(stdf['mode'])


for i,p in enumerate([p[0],pmean,pmode,p[999]]):

    X,Y,Z=meq.model(ARA,p)
    df_X=pd.DataFrame(X,columns=ARA)
    df_Y=pd.DataFrame(Y,columns=ARA)
    df_Z=pd.DataFrame(Z,columns=ARA)

    plt.subplot(4,3,(1+3*i))
    sns.heatmap(df_X, cmap="Reds")
    plt.subplot(4,3,(2+3*i))
    sns.heatmap(df_Y, cmap ='Blues')
    plt.subplot(4,3,(3+3*i))
    sns.heatmap(df_Z, cmap ='Greens')

plt.show()


X,Y,Z=meq.model(ARA,pmode)
plt.plot(X[:,0],label="DCoff")
plt.plot(X[:,3],label="AC1")
plt.plot(X[:,6],label="AC2")
plt.plot(X[:,9],label="DCon")
plt.plot([200,200],[0,1000],'--')

plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
'''

#####1indvs2ind
def plotdesnity1vs2():
    p2,df2= load('final','ACDC_X',parlist)
    parlist1=parlist.copy()
    del parlist1[7:9]
    p1,df1= load('12','ACDC_1ind',parlist1)

    namelist=[]
    for i,par in enumerate(parlist1):
        namelist.append(par['name'])
        
    parl = np.append(namelist,'dist')
    index=1
    for i,name in enumerate(parl):
        plt.subplot(4,4,index)
        plt.tight_layout()

        sns.kdeplot(df1[name],bw_adjust=.8,label='X_1ind')
        sns.kdeplot(df2[name],bw_adjust=.8,label='X_2ind')
            #plt.ylim(0,1)
        if i < (len(parl)-2):
            plt.xlim((parlist1[i]['lower_limit'],parlist1[i]['upper_limit']))
        index=index+1
        if index==5:       
            plt.legend(bbox_to_anchor=(1.05, 1))

        #sns.kdeplot(df['K_XZ'])
    plt.savefig("1vs2ind"+str(n[0])+"_compareplot.pdf", bbox_inches='tight')
    plt.show()

def ind1vs2indmeanandmode():
    p2,df2= load('final','ACDC_X',parlist)
    df2=df2.drop(columns=['K_ARAY', 'n_ARAY'])
    mean_df2=np.mean(df2)
    sd_df2=np.std(df2)
    mode_df2=stats.mode(df2)[0][0]
    parlist1=parlist.copy()
    del parlist1[7:9]
    p1,df1= load('12','ACDC_1ind',parlist1)
    mean_df1=np.mean(df1)
    sd_df1=np.std(df1)
    mode_df1=stats.mode(df1)[0][0]

    namelist=[]
    for i,par in enumerate(parlist1):
        namelist.append(par['name'])            
    parl = np.append(namelist,'dist')
    # set width of bars
    barWidth = 0.30
    # Set position of bar on X axis
    r1 = np.arange(len(parl))
    plt.bar((r1+barWidth*0),mean_df1,yerr=sd_df1, capsize=2,width=barWidth, label="1ind")
    plt.bar((r1+barWidth*1),mean_df2,yerr=sd_df2, capsize=2,width=barWidth, label="2ind")
    plt.xlabel('par', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(parl))], parl)
    plt.legend()
    plt.show()
    plt.bar((r1+barWidth*0),mode_df1,width=barWidth, label="1ind")
    plt.bar((r1+barWidth*1),mode_df2,width=barWidth, label="2ind")
    plt.xlabel('par', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(parl))], parl)
    plt.legend()
    plt.show()






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


#chose parameter
def bifurcation(parUsed=None):
    p,df= load('final','ACDC_X2',parlist)
    #parUsed=par0
    if parUsed == None:
        parUsed=p[0]
    ARA=np.logspace(-4.5,-2.,20,base=10)
    ss=meq.findss(ARA[0],parUsed)[0]
    #print(ss)
    init=[ss[0],ss[1],ss[2]]
    X,Y,Z=meq.model(ARA,parUsed,totaltime=100,init=init)
    df_X=pd.DataFrame(X[500:],columns=ARA)
    sns.heatmap(df_X, cmap="Reds", norm=LogNorm())
    plt.show()

    xss,yss,zss = calculateSScurve(ARA,parUsed)



    maxX=[]
    minX=[]
    maxY=[]
    minY=[]
    maxZ=[]
    minZ=[]
   # X,Y,Z=meq.model(ARA,parUsed,totaltime=400)
    delta=10e-5
    for i in np.arange(0,len(ARA)):
        min_x=[np.nan,np.nan,np.nan]
        max_x=[np.nan,np.nan,np.nan]
        ss=meq.findss(ARA[i],parUsed)
        for si,s in enumerate(ss):
            init=[s[0]+delta,s[1]+delta,s[2]+delta]
            X,Y,Z=meq.model(ARA,parUsed,totaltime=100,init=init)
           # print(max(X[200:,i]))

            max_x[si]=max(X[200:,i])
            min_x[si]=min(X[200:,i])

          



        maxX.append(max_x)
        minX.append(min_x)

        
       # minX.append(min(X[200:,i]))
        maxY.append(max(Y[200:,i]))
        minY.append(min(Y[200:,i]))
        maxZ.append(max(Z[200:,i]))
        minZ.append(min(Z[200:,i]))
    plt.subplot(3,1,1)
    plt.plot(ARA,xss,'--o')
    plt.plot(ARA,maxX,'-b')
    plt.plot(ARA,minX,'-g')
    #plt.fill_between(ARA,maxX,minX,alpha=0.2,facecolor='red')
    plt.yscale("log")
    plt.xscale("log")
    plt.subplot(3,1,2)
    plt.plot(ARA,yss,'--b')
   # plt.plot(ARA,maxY,'-b')
   # plt.plot(ARA,minY,'-b')
   # plt.fill_between(ARA,maxY,minY,alpha=0.2,facecolor='blue')
    plt.yscale("log")
    plt.xscale("log")
    plt.subplot(3,1,3)
    plt.plot(ARA,zss,'--g')
  #  plt.plot(ARA,maxZ,'-g')
  #  plt.plot(ARA,minZ,'-g')
  #  plt.fill_between(ARA,maxZ,minZ,alpha=0.2,facecolor='green')
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


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

                    maximaStability = abs(maxValues[-2]-minValues[-2])-(maxValues[-3]-minValues[-3])
                    if maximaStability > 0.01:
                        print("limit cycle not achieved for ARA["+str(ai)+"]:" + str(a) + " at st.s:"+ str(s))

    return M,m





def bifurcation_plot(n,filename):
    p,df= load(n,filename,parlist)
    ARA=np.logspace(-4.5,-2.,200,base=10)
    un,st,osc=calculateSS(ARA,p[1])
    M,m=getlimitcycle(ARA,osc,p[1],tt=500)
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
    plt.show()


#bifurcation(p[1])
