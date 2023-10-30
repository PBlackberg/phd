models_cmip6 = [         # Models ordered by change in temperature with warming (similar models removed)
    'INM-CM5-0',         # 1
    'IITM-ESM',          # 2
    'FGOALS-g3',         # 3    
    'INM-CM4-8',         # 4                                
    'MIROC6',            # 5                                      
    'MPI-ESM1-2-LR',     # 6                         
    'KIOST-ESM',         # 7    
    'BCC-CSM2-MR',       # 8        
    'GFDL-ESM4',         # 9    
    'MIROC-ES2L',        # 10   
    'NorESM2-LM',        # 11      
    'NorESM2-MM',        # 12                                      
    'MRI-ESM2-0',        # 13                                  
    'GFDL-CM4',          # 14      
    'CMCC-CM2-SR5',      # 15                
    'CMCC-ESM2',         # 16                                    
    'NESM3',             # 17     
    'ACCESS-ESM1-5',     # 18   
    'CNRM-ESM2-1',       # 19   
    'EC-Earth3',         # 20
    'CNRM-CM6-1',        # 21  
    'CNRM-CM6-1-HR',     # 22   
    'KACE-1-0-G',        # 23            
    'IPSL-CM6A-LR',      # 24
    'ACCESS-CM2',        # 25   
    'TaiESM1',           # 26                            
    'CESM2-WACCM',       # 27   
    'CanESM5',           # 28  
    'UKESM1-0-LL',       # 29  
    ]


models_excluded = []
models_exclude = ['GFDL-ESM4', 'CMCC-CM2-SR5', 'CNRM-ESM2-1']
models_cmip6 = list(filter(lambda x: x not in models_exclude, models_cmip6))
models_excluded.extend([m for m in            models_exclude if m not in models_excluded])





print(models_excluded)



