
#list for ACDC
parlist = [ # list containing information of each parameter
    #first node X param
    {'name' : 'K_ARAX', 'lower_limit':-5.0,'upper_limit':0.0}, #in log
    {'name' : 'n_ARAX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XY','lower_limit':-3.0,'upper_limit':0.0},
    {'name' : 'n_XY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XZ','lower_limit':-3.0,'upper_limit':0.0},
    {'name' : 'n_XZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_X','lower_limit':0.1,'upper_limit':1.0},
    {'name' : 'alpha_X','lower_limit':0.0,'upper_limit':1.0},
    {'name' : 'delta_X','lower_limit':0.99,'upper_limit':1.0},


    #Seconde node Y param
    {'name' : 'K_ARAY', 'lower_limit':-5.0,'upper_limit':0.0}, #in log
    {'name' : 'n_ARAY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_YZ','lower_limit':-3.0,'upper_limit':0.0},
    {'name' : 'n_YZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_Y','lower_limit':0.1,'upper_limit':1.0},
    {'name' : 'alpha_Y','lower_limit':0.0,'upper_limit':1.0},
    {'name' : 'delta_Y','lower_limit':0.99,'upper_limit':1.0},


    #third node Z param
    {'name' : 'K_ZX','lower_limit':-3.0,'upper_limit':0.0},
    {'name' : 'n_ZX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta_Z','lower_limit':0.1,'upper_limit':1.0},
    {'name' : 'alpha_Z','lower_limit':0.0,'upper_limit':1.0},
    {'name' : 'delta_Z','lower_limit':0.99,'upper_limit':1.0},
]


ARA=np.logspace(-4.5,-2.,8,base=10) #for ACDC
#ARA = np.array([0]) #for rep
