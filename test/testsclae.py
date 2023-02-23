import basicFunction as bf
import pandas as pd
import matplotlib.pyplot as plt

pricedata = bf.getdata(interval='0915')

#RV
RV = {}
for i in range(len(pricedata)):
    RV[i] = bf.calculRV(price_data=pricedata[i],interval='0915')
    RV[i].index = pd.to_datetime(RV[i].index,format='%Y%m%d')

result = pd.DataFrame()

for i in range(len(RV)):
    result = pd.concat([result,RV[i]],axis=0)

result = result.sort_index(ascending=True,axis=0)
plt.plot(result)
plt.show()

