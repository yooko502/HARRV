import basicFunction as bf
import warnings
import harmodel
from har_svrmodel import HARSVRmodel
warnings.filterwarnings('ignore')

price_data = bf.getdata(interval='0910')
price_data = bf.concatRV(price_data)
RV_positive, RV = bf.calculRV(price_data=price_data, type='RV+')

harx = harmodel.HARX(y=RV,other_y=RV_positive, lags=[1,5,22], other_type='RV+')

harsvr = HARSVRmodel(y=RV, other_y=RV_positive, lags=[1,5,22], other_type='RV+', num_generations=1, modeltype=1, useData='test')