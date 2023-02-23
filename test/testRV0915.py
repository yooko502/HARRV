import basicFunction as bf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

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

fig,(ax1,ax2) = plt.subplots(2,1,figsize = (10,8))
plot_acf(result,lags = 20,ax=ax1)
plot_pacf(result,lags = 20,ax = ax2)

plt.show()
'''
HAR-SVR model

RV^hat = alpha * RV_HAR +(1-alpha) * RV_SVR
用alpha來平衡线性和非线性预测。。。有人做过了。
但是alpha的确定方式没有人用过GA。
还是得RV_HAR+epsilon
epsilon的预测自变量不好确定


'''


