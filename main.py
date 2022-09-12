'''

Do call the difference analysis of ACDC


'''

import numpy as np
import matplotlib.pyplot as plt
from plot import *

import seaborn as sns

import statistics


from scipy.stats import gaussian_kde
from matplotlib import colors






'''
================================================================================

SET UP 

===============================================================================
'''
inch_cm =2.54


filename="ACDC_X_2ind_new_inithigh"#1ind"
n=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','final']

#filename="ACDC_X21ind"
#n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
#n=['7']

sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
#sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/ACDC_light/'+filename)
import model_equation as meq
parlist=meq.parlist

if os.path.isdir(filename+'/plot') is False: ## if 'smc' folder does not exist:
    os.mkdir(filename+'/plot') ## create it, the output will go there

'''
================================================================================

Analyse 1 : heatmap

===============================================================================
'''

n=['1','5','10','final']
heatmap_allroound(meq.ARA,filename,meq.parlist,n,meq)
plt.savefig(filename+"/plot/_heatmap.pdf", dpi=300)
#plt.show()




'''
================================================================================

Analyse 2 : par Space

===============================================================================
'''

#PARPLOT

n=['final']
p, df= load(n[0],filename,meq.parlist)
par_plot(df,parlist)
plt.savefig(filename+"/plot/_parspace.pdf", dpi=300)
#plt.show()




'''
================================================================================

Analyse 3 : par stats

still need to do

===============================================================================
'''
'''
n='final'
p, df= load(n,filename,meq.parlist)
barplot(df,parlist)
plt.savefig(filename+"_barplot.pdf", dpi=300)
plt.show()
'''

'''
================================================================================

Analyse 4 : 1 vs 2 ind comparisons

still need to do

===============================================================================
'''







'''
================================================================================

Analyse 5 : Bifurcation

===============================================================================
'''



'''
##########################################single plot
n='final'
par, df= load(n,filename,meq.parlist)
ARA=meq.ARA
ARA=np.logspace(-5,-1.,10,base=10)
i=0
bifuplot(ARA,filename,par[i],meq,i)
plt.show()
'''
'''
############################################grid
n='final'
par, df= load(n,filename,meq.parlist)
pp=np.arange((25))
s=int(np.sqrt(len(pp)))
fig,axs= plt.subplots(s,s, figsize=(s,s))
ARA=meq.ARA
x=0
y=0
for i,ni in enumerate(pp):
    bifuplot_grid(axs,x,y,ARA,filename,p[ni],meq,ni)
    axs[x,y].text(ARA[-2],10e-5, ("p "+str(ni)),fontsize=8,ha='center', va='center')
    niceaxis(axs,x,y,ni,ni,pp,pp,6)
    x+=1
    if x==s:
        x=0
        y+=1

plt.show()

'''

##############################################chek ACDC behavior

'''
n='final'
par, df= load(n,filename,meq.parlist)
ARA=np.logspace(-4.5,-2.,40,base=10)
ARA=meq.ARA

#getBehaviorIndex(par,ARA,filename,meq) #calculated bifurcation style, to run it only once and save it after
#print("done")

f,l,h,b,c,t,o = loadBifurcation(filename)

print(f.size,l.size,h.size,b.size,c.size,t.size,o.size)
p=par
pi=t

if(pi.size==1):
    pi=[pi]

ms= int(np.round((np.sqrt(len(pi))+0.5)))
s=np.max((2,ms))

fig,axs= plt.subplots(s,s, figsize=(s,s))
ARA=meq.ARA
x=0
y=0
for i in pi:
    i=int(i)
    bifuplot_grid(axs,x,y,ARA,filename,p[i],meq,i)
    axs[x,y].text(ARA[-2],10e-5, ("p "+str(i)),fontsize=8,ha='center', va='center')
    niceaxis(axs,x,y,i,i,pi,pi,6)
    x+=1
    if x==s:
        x=0
        y+=1

plt.show()
'''


