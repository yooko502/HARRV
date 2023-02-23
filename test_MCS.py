import numpy as np
from arch.bootstrap import MCS
import pandas as pd

# load data
if __name__ == '__main__':

    y_real = pd.read_csv('data/test_data/0915_RV.csv', index_col=0)


def match_data(forecast, y):
    '''
    match the data
    :param forecast: the forecasting data
    :param y: the real data
    :return: the matched data
    '''

    y = y.shift(-1)[y.index.isin(forecast.index)].dropna()
    forecast = forecast[forecast.index.isin(y.index)]
    forecast_columns = forecast.columns
    forecast = forecast.rename(columns={col:''for col in forecast.columns})
    y = y.rename(columns={col:''for col in y.columns})

    return forecast, y, forecast_columns


def cal_error_mse(forecast, y):
    '''
    calculate the MSE error of forecasting for one day.
    :param forecast: the forecasting data
    :param y: the real data
    :return: the error
    '''
    # TODO：把data分为几个区块分别计算error function
    error = (forecast - y) ** 2

    return error


def cal_error_mae(forecast, y):
    '''
    calculate the MAE error of forecasting for one day.
    :param forecast: the forecasting data
    :param y: the real data
    :return: the error
    '''
    error = abs(forecast - y)

    return error


def cal_error_qlike(forecast, y):
    '''
    calculate the Q-LIKE error of forecasting for one day.
    :param forecast:
    :param y:
    :return:
    '''

    error = np.log(forecast) + y/forecast

    return error

def cal_error_mape(forecast, y):
    '''
    calculate the MAPE error of forecasting for one day.
    :param forecast: the forecasting data
    :param y: the real data
    :return: the error
    '''
    error = abs((forecast - y)/y)

    return error


def main(y_pred, y_r):
    # QLIK 报错 所以先去掉了
    forecast, y, forecast_columns= match_data(y_pred, y_r)
    error_mse = cal_error_mse(forecast, y)
    error_mae = cal_error_mae(forecast, y)
    # error_qlike = cal_error_qlike(forecast, y)
    error_mape = cal_error_mape(forecast, y)
    # TODO:这里会出现莫名其妙的index不对应的BUG在下面mcs_qlike.compute()的时候 在method = 'R'的时候 max的时候也会

    mcs_mse = MCS(error_mse, size=0.05, method='max')
    mcs_mae = MCS(error_mae, size=0.05, method='max')
    # mcs_qlike = MCS(error_qlike, size=0.05, method='max')
    mcs_mape = MCS(error_mape, size=0.05, method='max')

    mcs_mae.compute()
    mcs_mse.compute()
    # mcs_qlike.compute()
    mcs_mape.compute()

    # mcs_result = pd.concat([mcs_mse.pvalues, mcs_mae.pvalues, mcs_qlike.pvalues, mcs_mape.pvalues], axis=1)
    mcs_result = pd.concat([mcs_mse.pvalues, mcs_mae.pvalues, mcs_mape.pvalues], axis=1)
    # mcs_result.columns = ['mse', 'mae', 'Q-LIKE', 'mape']
    mcs_result.columns = ['mse', 'mae', 'mape']
    mcs_result.index = forecast_columns

    return mcs_result