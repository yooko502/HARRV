import numpy as np
from arch.bootstrap import MCS
import pandas as pd
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)

# load data
if __name__ == '__main__':

    y_real = pd.read_csv('data/test_data/0915_RV.csv', index_col=0)

interval_length = 22  # 计算error function的时候的时间间隔。

def q_like(y_pred, y):
    '''
    计算Q-LIKE的误差

    :param y_pred:
    :param y:
    :return: Q-like
    '''

    error = np.log(y_pred) + y/y_pred
    error = np.mean(error)

    return error


def calculate_num_of_interval(y):
    '''
    calculate the number of interval and remainder of the data.
    :param y: the data
    :return: the number of interval
    '''
    num_of_interval = y.shape[0] // interval_length
    remainder = y.shape[0] % interval_length

    return num_of_interval, remainder

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


def error_calculator(error_func):
    def wrapper(forecast, y):
        num_interval_forecast, remainder = calculate_num_of_interval(forecast)  # 获取传入数据的区块的个数
        num_interval_y, remainder = calculate_num_of_interval(y)

        if remainder > 0:
            num_of_loop = num_interval_forecast + 1
            errors = np.zeros(shape=(num_interval_forecast + 1, forecast.shape[1]))  # 存储所有的errors, size是区块数量x传入的模型的个数
        else:
            num_of_loop = num_interval_forecast
            errors = np.zeros(shape=(num_interval_forecast, forecast.shape[1]))

        for i in range(num_of_loop):  # 计算每个区块的error
            start_idx = i * interval_length
            end_idx = start_idx + interval_length

            for j in range(forecast.shape[1]):
                if end_idx > forecast.shape[0]:  # 判断end_idx是否超出了forecast的index范围
                    end_idx = forecast.shape[0]
                error = error_func(forecast.iloc[start_idx:end_idx, j], y[start_idx:end_idx])
                errors[i, j] = error

        return errors
    return wrapper


@error_calculator
def cal_error_mse(forecast, y):
    return mean_squared_error(forecast, y)


@error_calculator
def cal_error_mae(forecast, y):
    return mean_absolute_error(forecast, y)


def cal_error_qlike(forecast, y):
    '''
    calculate the Q-LIKE error of forecasting for one day.
    :param forecast:
    :param y:
    :return:
    '''

    num_interval_forecast, remainder = calculate_num_of_interval(forecast)  # 获取传入数据的区块的个数
    num_interval_y, remainder = calculate_num_of_interval(y)

    if remainder > 0:
        num_of_loop = num_interval_forecast + 1
        errors = np.zeros(shape=(num_interval_forecast + 1, forecast.shape[1]))  # 存储所有的errors, size是区块数量x传入的模型的个数
    else:
        num_of_loop = num_interval_forecast
        errors = np.zeros(shape=(num_interval_forecast, forecast.shape[1]))

    for i in range(num_of_loop):  # 计算每个区块的error
        start_idx = i * interval_length
        end_idx = start_idx + interval_length
        for j in range(forecast.shape[1]):
            if end_idx > forecast.shape[0]:  # 判断end_idx是否超出了forecast的index范围
                end_idx = forecast.shape[0]
            error = q_like(forecast.iloc[start_idx:end_idx, j], y[start_idx:end_idx])
            errors[i, j] = error

    return errors


@error_calculator
def cal_error_mape(forecast, y):
    return mean_absolute_percentage_error(forecast, y)


def main(y_pred, y_r):
    # QLIK 报错 所以先去掉了
    forecast, y, forecast_columns = match_data(y_pred, y_r)
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
