from har_svrmodel import HARSVRmodel

harsvr_tscv = HARSVRmodel(y = RV,lags=[1,5,22],useData='test',modeltype='tscv',\
                            num_generations=num_generations,\
                            gene_space=[{'low': 1e-6, 'high': 100}, {'low': 1e-2, 'high': 200}, {'low': 1e-6, 'high': 100}])