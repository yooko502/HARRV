import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit


def getdata(interval='1621'):
    '''
    从本地读取 2016-2021 TOPIX index high-frequency data.
    interval : 选取什么时间段的data,type : string ex. 2009-2015 = '0915',2016-2021 = '1621'
    group_index : 利用groupby给分组加上标签，为了前后一致，此处的标签使用sec
    2009-2010的data不是以秒为单位的所以暂时先放下
    timeflag：前场是1后场是2，用于对RV的数据进行噪声调整
    return : dataframe-type dataset
    '''

    def group_index(dataframe):
        dataframe['sec'] = range(dataframe.shape[0])
        return dataframe
    # TODO:data 路径改了 重新设定路径
    all_pathset_0915 = ['/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2009.csv',
               '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2010.csv',
               '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2011.csv',
               '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2012.csv',
               '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2013.csv',
               '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2014.csv',
               '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionData_0113_2015.csv']

    interval = interval
    if interval == '1621':
        dataset = {}
        headlist = ['ID', 'date', 'hour', 'min', 'sec', 'microsec', \
                    'price', 'volume', 'timeflag']
        droplist = ['ID', 'microsec', 'volume']
        i = 0
        pathset = ['/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2016.csv',
                   '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2017.csv',
                   '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2018.csv',
                   '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2019.csv',
                   '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2020.csv',
                   '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2021.csv',
                   '/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/data/pricedata/TickVisionUHFD_0113_2022.csv']
        for path in pathset:
            dataset[i] = pd.read_csv(path, header=None)
            dataset[i].columns = headlist
            dataset[i] = dataset[i].drop(columns=droplist)
            i = i + 1

        return dataset

    elif interval == '09':
        dataset = {}
        headlist = ['date', 'time', 'price', 'volume']
        droplist = ['volume']
        i = 0
        pathset = all_pathset_0915[0]
        for path in pathset:
            dataset[i] = pd.read_csv(path, header=None)
            dataset[i].columns = headlist
            dataset[i] = dataset[i].drop(columns=droplist)
            dataset[i] = dataset[i].groupby(['date', 'time']).apply(group_index)
            dataset[i]['timeflag'] = np.where(dataset[i]['time'] < 120000, 1, 2)
            i = i + 1

        return dataset

    elif interval == '0915':
        dataset = {}
        headlist = ['date', 'time', 'price', 'volume']
        droplist = ['volume']
        i = 0
        pathset = all_pathset_0915
        for path in pathset:
            dataset[i] = pd.read_csv(path, header=None)
            dataset[i].columns = headlist
            dataset[i] = dataset[i].drop(columns=droplist)
            dataset[i] = dataset[i].groupby(['date', 'time']).apply(group_index)
            dataset[i]['timeflag'] = np.where(dataset[i]['time'] < 120000, 1, 2)
            i = i + 1

        return dataset
    elif interval == '0910':
        dataset = {}
        headlist = ['date', 'time', 'price', 'volume']
        droplist = ['volume']
        i = 0
        pathset = all_pathset_0915[0:2]
        for path in pathset:
            dataset[i] = pd.read_csv(path, header=None)
            dataset[i].columns = headlist
            dataset[i] = dataset[i].drop(columns=droplist)
            dataset[i] = dataset[i].groupby(['date', 'time']).apply(group_index)
            dataset[i]['timeflag'] = np.where(dataset[i]['time'] < 120000, 1, 2)
            i = i + 1

        return dataset


def getRVdata():
    '''
    使用老师给的 realized volatility data.

    :return: 1 min interval realized volatility data
    '''
    RVset = {}
    i = 0
    headlist = ['year', 'month', 'day', '1', '2', \
                '3', '4', 'daily RV']

    droplist = ['1', '2', '3', '4']
    '''
    只需要使用全体的Realized volatility
    '''

    pathset = ['data/RV/rv_2015_01_0113.csv',
               'data/RV/rv_2016_01_0113.csv',
               'data/RV/rv_2017_01_0113.csv',
               'data/RV/rv_2018_01_0113.csv',
               'data/RV/rv_2019_01_0113.csv',
               'data/RV/rv_2020_01_0113.csv',
               'data/RV/rv_2021_01_0113.csv']

    for path in pathset:
        RVset[i] = pd.read_csv(path, header=None)
        RVset[i].columns = headlist
        RVset[i] = RVset[i].drop(droplist)
        i = i + 1

    return RVset


def dataselect(data, DeltaT='1min', interval='0915'):
    '''
    A function for selecting data from original data.
    比如选择30秒间隔的data或者1分钟间隔的data etc...
    使用每分钟数据的时候使用的是每分钟的开始的数据，而并非每分钟的终值。
    :param data: 传入的data dataframe
    :param DeltaT: delta t of RV. Default : 5 mins.
    :param interval: 选取什么时间段的data,type : string ex. 2009-2015 = '0915',2016-2021 = '1621'
    :return: selected data by delta t.
    '''
    DeltaT = DeltaT
    if interval == '1621':

        dTinterval = ['15s', '30s', '1min', '5mins']
        if DeltaT not in dTinterval:
            raise ValueError('Error, wrong delta t.Delta T should be {}'.format(dTinterval))
        if DeltaT == '15s':
            dataset = data[(data.sec + 1) % 15 == 0]
            return dataset
        if DeltaT == '30s':
            dataset = data[(data.sec + 1) % 30 == 0]
            return dataset
        if DeltaT == '1min':
            dataset = data[(data.sec + 1) % 60 == 0]
            return dataset
        if DeltaT == '5mins':
            dataset = data[(data.sec + 1) % 60 == 0][data['min'] % 5 == 0]
            return dataset

    '''
    因为数据内每分钟交易数量不一致的问题，现在只使用一分钟的交易数据来进行计算。
    
    '''
    if interval == '0915':
        dTinterval = ['1min']
        if DeltaT not in dTinterval:
            raise ValueError('Error, wrong delta t.Delta T should be {}'.format(dTinterval))

        if DeltaT == '15s':
            pass

        if DeltaT == '30s':
            pass

        if DeltaT == '1min':
            data = data[data.sec == 0]
            return data

        if DeltaT == '5mins':
            pass


'''
calculate RV+ and RV- and SJ
'''


def calculRVplusminus(logreturn_positive, logreturn_minus, i):

    rv_positive = i * logreturn_positive.groupby('date').sum()

    rv_minus = i * logreturn_minus.groupby('date').sum()
    sj = rv_positive - rv_minus

    rv_positive.columns = ['RV+']
    rv_minus.columns = ['RV-']
    sj.columns = ['SJ']

    return rv_positive, rv_minus, sj

#TODO：测试新方法的正确性
'''计算的方法不一定有问题，但是要写09-15的data和16-21的data的区分'''


def calculRV(price_data, type = None,window='daily', DeltaT='1min', night=True, interval='0915', scale=True):
    '''
    A funtion for calculating daily or monthly or weekly Realized Volatility.

    RV的计算参考 CORSI (2009)
    equation 3 & 4

    window_set :需要计算的RV的类型，daily是日RV,weekly是最近5天的RV的平均，
    monthly是最近22天RV的平均。

    log_pr_data : Logarithm price.对数价格
    log_return : Logarithm return.对数收益率
    :param price_data: high-frequency price data.
    :param window: Default daily.
    :param DeltaT: delta t of RV. Default 5 mins.
    :param night: 是否包含隔夜交易，第二天的开盘价减去第一天的收盘价
    :param interval: 选取什么时间段的data,type : string ex. 2009-2015 = '0915',2016-2021 = '1621'
    :param scale: 是否对data进行放缩 默认True，默认放缩为1万倍

    log_pr_data : Logarithm price.对数价格
    log_return : Logarithm return.对数收益率
    price_data : the data which include any element.
    logreturn_data : the data which only include date and square of log return
    logprice_data: the data which only include date and log price

    RV: sqrt(sum of daily realized volatility's square), realized volatility

    return: daily realized volatility or weekly or monthly.
    note:只能计算年内的weekly monthly 的RV,无法计算跨年度的weekly monthly RV.

    :return:
    '''
    DeltaT = DeltaT
    dTinterval = ['15s', '30s', '1min', '5mins']
    logreturn_data = pd.DataFrame()
    logprice_data = pd.DataFrame()
    logreturn_minus = pd.DataFrame()
    logreturn_positive = pd.DataFrame()

    if DeltaT not in dTinterval:
        raise ValueError('Error, wrong delta t. DeltaT should be {}'.format(dTinterval))
    if night == False:

        price_data = dataselect(data=price_data, DeltaT=DeltaT, interval=interval)
        price_data['logprice'] = np.log(price_data['price'])
        logprice_data['date'] = price_data['date']
        logprice_data['logprice'] = price_data['logprice']
        logreturn_data['date'] = price_data['date']
        logreturn_data['logreturn'] = np.square \
            (logprice_data.groupby('date').diff())

    elif night == True:

        price_data = dataselect(data=price_data, DeltaT=DeltaT, interval=interval)
        price_data['logprice'] = np.log(price_data['price'])

        logprice_data['logprice'] = price_data['logprice']
        logreturn_data['date'] = price_data['date']
        logreturn_data['logreturn'] = np.square(logprice_data.diff())
        '''
        Put the positive value and negative value of logorice_data.diff() into logreturn_positive and logreturn_minus
        '''
        # 把logreturn进行平方
        logreturn_positive['date'] = price_data['date']
        logreturn_positive['logreturn'] = np.where(logprice_data.diff() > 0, logprice_data.diff(), 0) ** 2

        logreturn_minus['date'] = price_data['date']
        logreturn_minus['logreturn'] = np.where(logprice_data.diff() < 0, logprice_data.diff(), 0) ** 2

    else:
        raise TypeError('Error, parameter night should be boolean.')

    '''
    calculate daily Realized volatility
    '''
    # 选择RV的放缩的倍数
    if scale:
        i = 10000
    elif not scale:
        i = 1
    else:
        raise TypeError('Error, parameter scale should be boolean.')

    # 计算RV
    RV = i * logreturn_data.groupby('date').sum()

    if type is not None:

        rv_positive, rv_minus, sj = calculRVplusminus(logreturn_positive, logreturn_minus, i)

        # 移到了计算的函数中
        # RV.columns = ['Daily RV']
        # rv_positive.columns = ['RV+']
        # rv_minus.columns = ['RV-']
        # sj.columns = ['SJ']

        output_mapping = {
            'RV+': (rv_positive, RV),
            'RV-': (rv_minus, RV),
            'SJ': (sj, RV),
        }
        return output_mapping[type]

    window = window
    window_set = ['daily', 'monthly', 'weekly']

    if window not in window_set:
        raise TypeError('Window type error,window should be {}'.format(window_set))
    if window == 'daily':
        RV.columns = ['Daily RV']
        return RV
    elif window == 'weekly':
        RV.columns = ['Weekly RV']
        return RV.rolling(windows=5).mean()
    elif window == 'monthly':
        RV.columns = ['Monthly RV']
        return RV.rolling(window=22).mean()


def calculateRV(RVdata, window):
    '''
    通过daily RV来计算其他的RV
    :param number: 需要额外计算的RV的数量
    :param RVdata: 已经计算好的 daily RV ,dataframe
    :param window: 需要计算的RV的平均的天数,int or tuple
    :return: 得到的RVdata
    '''

    number = len(window)
    RVlist = pd.DataFrame()
    RVlist['daily RV'] = RVdata
    for i in range(number):
        RVlist['{}days RV'.format(window[i])] = RVdata.rolling(window=window[i]).mean()

    return RVlist


def concatRV(RVlist):
    '''
    功能：将多年的RV进行拼接，用于进行连续的预测
    RVlist: type : dict
    :return:拼接好的 list type：dataframe
    '''
    result = pd.DataFrame()

    for i in range(len(RVlist)):
        result = pd.concat([result, RVlist[i]], axis=0)

    #result = result.sort_index(ascending=True, axis=0)

    return result


def adjustRV(dailyRV):
    pass
