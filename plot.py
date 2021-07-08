#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import abc_smc
import model_equation as meq
import statistics

from collections import Counter

number="1"



path = 'smc_'+'/pars_' + number + '.out'
dist_path = 'smc_'+'/distances_' + number + '.out'

raw_output= np.loadtxt(path)
dist_output= np.loadtxt(dist_path)
df = pd.DataFrame(raw_output, columns = ['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ', 'beta_X','alpha_X','delta_X',
                                          'K_ARAY','n_ARAY','K_YZ','n_YZ', 'beta_Y','alpha_Y','delta_Y',
                                          'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z'])
df['dist']=dist_output



# print(df[df['dist']==min(df['dist'])])

p_min=df[df['dist']==min(df['dist'])]

p=[p_min['K_ARAX'].tolist()[0],p_min['n_ARAX'].tolist()[0],p_min['K_XY'].tolist()[0],p_min['n_XY'].tolist()[0],p_min['K_XZ'].tolist()[0],p_min['n_XZ'].tolist()[0], 
p_min['beta_X'].tolist()[0],p_min['alpha_X'].tolist()[0],p_min['delta_X'].tolist()[0],
p_min['K_ARAY'].tolist()[0],p_min['n_ARAY'].tolist()[0],p_min['K_YZ'].tolist()[0],p_min['n_YZ'].tolist()[0], p_min['beta_Y'].tolist()[0],
p_min['alpha_Y'].tolist()[0],p_min['delta_Y'].tolist()[0],
p_min['K_ZX'].tolist()[0],p_min['n_ZX'].tolist()[0], p_min['beta_Z'].tolist()[0],p_min['alpha_Z'].tolist()[0],p_min['delta_Z'].tolist()[0]]


p_final=abc_smc.pars_to_dict(p)
meq.plot(meq.ARA,p_final)


sns.pairplot(df[['K_ARAX','n_ARAX','K_XY','n_XY','K_XZ','n_XZ', 'beta_X','alpha_X','delta_X',
                                          'K_ARAY','n_ARAY','K_YZ','n_YZ', 'beta_Y','alpha_Y','delta_Y',
                                          'K_ZX','n_ZX', 'beta_Z','alpha_Z','delta_Z']], kind='kde')
plt.savefig('par_plot.pdf', bbox_inches='tight')





