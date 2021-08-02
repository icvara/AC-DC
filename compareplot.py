#compare parameter between abc-smc
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats

#filename=["ACDC_X","ACDC_Y","ACDC_Z","ACDC_all"]
filename=['ACDC_1ind']
#filename=['ACDC_X','ACDC_1ind']
#n=['final']
n=['1','2','3','4','5','6','7','8','9','10','11','12','final']#'13','14','15','final']
#n=['1','2','3','4','5','6','7','8','9','10','11','12','13','14']

path='C:/Users/Administrator/Desktop/Modeling/AC-DC/'
sys.path.insert(0, path + filename[0])
import model_equation as meq
  
parlist=meq.parlist
namelist=[]
for i,par in enumerate(parlist):
    namelist.append(parlist[i]['name'])

def pars_to_dict(pars):
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
        p0=pars_to_dict(p0)
        p.append(p0)    
    return p, df


def get_stats(filename,namelist):
    stats_df = pd.DataFrame( columns = ['par','method','mean','sd'])#,'mode'])
    parl = np.append(namelist,'dist')
    for fi,fnm in enumerate(filename):
        p,df= load(n[0],fnm,parlist)
        mean=np.mean(df).tolist()
        sd=np.std(df).tolist()
        mode=stats.mode(df)[0][0]
        new_row={'par':parl,'method':[fnm]*len(parl),'mean':mean,'sd':sd,'mode':mode}
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


def plot_compare(filename,namelist):
    parl = np.append(namelist,'dist')
    index=1
    for i,name in enumerate(parl):
        plt.subplot(4,4,index)
        plt.tight_layout()
        for fi,fnm in enumerate(filename):
            p,df= load(n[0],fnm,parlist)
            sns.kdeplot(df[name],bw_adjust=.8,label=fnm)
        #plt.ylim(0,1)
        if i < (len(parl)-2):
            plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
        index=index+1
        if index==5:       
            plt.legend(bbox_to_anchor=(1.05, 1))

    #sns.kdeplot(df['K_XZ'])
    plt.savefig(str(filename)+str(n[0])+"_compareplot.pdf", bbox_inches='tight')
    plt.show()


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
        #    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

