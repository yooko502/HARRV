import pandas as pd

import basicFunction as bf
#from arch.univariate import HARX
from harmodel import HARX

abc = bf.getdata('0915')

RV = bf.calculRV(abc[0])
RV.index = pd.to_datetime(RV.index,format='%Y%m%d')
harx = HARX(RV,lags=[1,5,22])

harx1 = harx.fit()