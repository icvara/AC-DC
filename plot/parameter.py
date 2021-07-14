#list for ACDC

parlist = [ # list containing information of each parameter
    #first node X param
    {'name' : 'K_ARAX', 'lower_limit':-5.0,'upper_limit':0.0}, #in log
    {'name' : 'n_ARAX','lower_limit':1.5,'upper_limit':2.5},
    {'name' : 'K_XY','lower_limit':0.001,'upper_limit':0.02},
    {'name' : 'n_XY','lower_limit':1.5,'upper_limit':2.5},
    {'name' : 'K_XZ','lower_limit':0.001,'upper_limit':0.02},
    {'name' : 'n_XZ','lower_limit':1.5,'upper_limit':2.5},
    {'name' : 'beta_X','lower_limit':1.0,'upper_limit':1.05},
    {'name' : 'alpha_X','lower_limit':0.0,'upper_limit':0.5},
    {'name' : 'delta_X','lower_limit':0.95,'upper_limit':1.0},


    #Seconde node Y param
    {'name' : 'K_ARAY', 'lower_limit':-5.0,'upper_limit':0.0}, #in log
    {'name' : 'n_ARAY','lower_limit':1.5,'upper_limit':2.5},
    {'name' : 'K_YZ','lower_limit':0.001,'upper_limit':0.02},
    {'name' : 'n_YZ','lower_limit':1.5,'upper_limit':2.5},
    {'name' : 'beta_Y','lower_limit':1.0,'upper_limit':1.05},
    {'name' : 'alpha_Y','lower_limit':0.0,'upper_limit':0.5},
    {'name' : 'delta_Y','lower_limit':0.95,'upper_limit':1.0},


    #third node Z param
    {'name' : 'K_ZX','lower_limit':0.001,'upper_limit':0.02},
    {'name' : 'n_ZX','lower_limit':1.5,'upper_limit':2.5},
    {'name' : 'beta_Z','lower_limit':1.0,'upper_limit':1.05},
    {'name' : 'alpha_Z','lower_limit':0.0,'upper_limit':0.5},
    {'name' : 'delta_Z','lower_limit':0.95,'upper_limit':1.0},
]


ARA=np.logspace(-4.5,-2.,10,base=8) #for ACDC