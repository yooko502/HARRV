import pandas as pd
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
import basicFunction as bf
import seaborn as sns
import matplotlib.pyplot as plt
from har_svrmodel import HARSVRmodel
from harmodel import HARX
import warnings
import time
import test_MCS as tm
import os
import shutil
warnings.filterwarnings("ignore")
## TODO： 增加几行代码，让保存结果的时候可以存在不同的文件夹里 让保存的结果不那么乱


'''
Note：暂时没有什么太大的问题和缺失的功能，等待后续用全部的data再进行测试。
'''

'''
a decorator to calculate the time used for each run.
'''


def timer(func):

    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print('time used:{} seconds'.format(end-start))
        return result, end-start
    return wrapper

'''
Use HAR model to predict RV.
'''


@timer
def run_har_model(RV, observation, other_y=None, other_type=None):
    '''
    use HAR model to predict RV.
    :param other_y: other risk measure data except RV.
    :param other_type: the type of other risk measure. like RV+,RV-,SJ,etc.
    :param RV:realized volatility
    :param observation:
    :return:
    '''
    result = pd.DataFrame()
    for i in range(0, len(RV)-observation):

        RV_used = RV.iloc[i:i+observation+1, :]
        if other_y is not None:
            other_y_used = other_y.iloc[i:i+observation+1, :]
        else:
            other_y_used = None
        harx = HARX(y=RV_used, other_y=other_y_used, other_type=other_type, lags=[1, 5, 22], observation=observation)
        harx_fit = harx.fit()
        predict_result = harx_fit.predict(harx.out_predictor).to_frame()
        if other_type is None:
            risk_measure_1 = 'RV'
        else:
            risk_measure_1 = other_type
        predict_result.columns = ['Daily {}'.format(risk_measure_1)]
        result = pd.concat([result, predict_result], axis=0)
        del harx
    return result


def calculate_har_error(RV, har_result, measure, run_type, cross_validation):
    '''
    calculate the error of har model.
    :param RV: realized volatility real value.
    :param har_result: forecasted realized volatility of har model.
    :param measure: risk measure. like RV-,RV+,SJ,etc.
    :param run_type: test or 0910 or 0915 ,etc.
    :param cross_validation: cross validation method of genetic algorithm.
    :return:
    '''
    RV = RV.shift(-1)[RV.index.isin(har_result.index)].dropna()
    har_result = har_result[har_result.index.isin(RV.index)]
    errors = calculate_errors(har_result, RV)
    risk_measure = measure
    errors.to_csv('Result/{}/{}/{}/{}_har_error.csv'.format(risk_measure, run_type, cross_validation, run_type))

'''
run har svr model.
'''


@timer
def run_har_svr(RV, observation, model_type, times, num_generations, cross_validation, other_y=None, other_type=None):
    '''
    run har svr model.
    forecasted method: one day ahead after observation.
    :param RV: realized volatility real value.
    :param observation: number of observed data.
    :param model_type: 1 or 2 or 3.
    :param times: for recording the number of runs.
    :param num_generations: number of generations for genetic algorithm.
    :param cross_validation: cross_validation method for genetic algorithm's fitness function.
    :param other_y: other risk measure data except RV.
    :param other_type: risk measure type. like RV+,RV-,SJ,etc.
    :return: har-svr model's forecasted realized volatility for days after observation.
    '''
    result = pd.DataFrame()
    for i in range(0,len(RV)-observation):
        print('*'*50)
        start = time.time()
        print(f'run {i+1}th dataset for model {model_type}, risk measure {other_type}')
        print('running ...')
        RV_used = RV.iloc[i:i+observation+1, :]
        if other_y is not None:
            other_y_used = other_y.iloc[i:i+observation+1, :]
        else:
            other_y_used = None
        harsvr = HARSVRmodel(y=RV_used, other_y=other_y_used, other_type=other_type, lags=[1,5,22], useData='test',modeltype=model_type,\
                             num_generations=num_generations,\
                             gene_space=[{'low': 1e-6, 'high': 100}, {'low': 1e-2, 'high': 200}, {'low': 1e-6, 'high': 100}],\
                             observation=observation, Methodkeys=cross_validation)

        a = harsvr.predict()
        result = pd.concat([result,a],axis=0)
        print(f'{i+1}th dataset for model {model_type} is done. Used {time.time()-start} seconds. In {times+1}th runs. Risk measure {other_type}')
        print('*'*50)
        del harsvr
    return result


'''
run run_har_svr() for each model_type n times and save the result in result_all.
then calculate the mean and variance each day.
'''


def run_n_times(n, model_type, observation, num_generations, cross_validation, other_y=None, other_type=None):
    '''
    run run_har_svr() for each model_type n times and save the result in result_all.
    :param n: number of runs.
    :param model_type: model type. 1 or 2 or 3.
    :param observation: number of observed data.
    :param num_generations: number of generations for genetic algorithm.
    :param cross_validation: cross_validation method for genetic algorithm's fitness function.
    :param other_y: other risk measure data except RV.
    :param other_type: risk measure type. like RV+,RV-,SJ,etc.
    :return:
    '''
    result_all = pd.DataFrame()
    time_used = pd.DataFrame()
    statistics = {'mean': ..., 'var': ...}
    maxmum_val = pd.DataFrame()
    minmum_val = pd.DataFrame()

    start = time.time()
    for j in range(n):
        #TODO：print 一些东西增加监听功能
        print('*'*50)
        print(f'run {j+1} times for model {model_type} on risk measure {other_type}')
        result_all[j], time_used[j] = run_har_svr(RV=RV, other_y=other_y, other_type=other_type, \
                                                  observation=observation, model_type=model_type,times=j,\
                                                  num_generations=num_generations, cross_validation=cross_validation)
        result_all[j].columns = ['Number of times {}'.format(j)]
        print('*'*50)
        print(f'{j+1} times for model {model_type} is done. Risk measure {other_type}')
        end = time.time()
        print(f'already used {end-start} seconds for model {model_type} on risk measure {other_type}')
    # calculate the mean and variance of each day of n times run.

    maxmum_val['model'.format(model_type)] = result_all.max(axis=1)
    minmum_val['model'.format(model_type)] = result_all.min(axis=1)
    statistics['mean'] = result_all.mean(axis=1)
    statistics['mean'].index = result_all.index
    statistics['var'] = result_all.var(axis=1)
    statistics['var'].index = result_all.index

    return result_all, statistics, maxmum_val, minmum_val


result_all = pd.DataFrame()
time_used = pd.DataFrame()

'''
use seaborn.set() to set the style of the figure.
plot statistics result mean, maximum value, minimum value, RV by different color in one figure for each model_type.
and save the figure in Result folder.
'''


def plot_result(statistics_result, maximum_val, minimum_val, RV, model_type, har_result, measure, run_times, num_generations, cross_validation):
    # TODO：把最大值到最小值中间的区域填充颜色并且设定透明度
    '''
    plot statistics result mean, maximum value, minimum value, RV by different color in one figure for each model_type.
    :param statistics_result: mean and variance of forecasted realized volatility for days after observation.
    :param maximum_val: maximum value of forecasted realized volatility for days after observation.
    :param minimum_val: minimum value of forecasted realized volatility for days after observation.
    :param RV:
    :param model_type:
    :param har_result:
    :param measure: risk measure type. like RV+,RV-,SJ,etc.
    :param run_times:
    :param num_generations:
    :param cross_validation:
    :return:
    '''
    sns.set()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('risk measure {} HAR SVR model{},cross_validation {}, with run times {},and number of iterations{}'.\
                 format(measure, model_type, cross_validation, run_times, num_generations))
    ax.set_xlabel('Date')
    ax.set_ylabel('Realized Volatility')
    ax.plot(har_result, label='HAR Model')
    ax.plot(statistics_result['mean'], label=f'HAR-SVR Model{model_type-1}')

    # ax.plot(maximum_val, label='max')
    # ax.plot(minimum_val, label='min')
    RV = RV.shift(-1)[RV.index.isin(statistics_result['mean'].index)]
    ax.plot(RV, label='Real Value')
    ax.legend()
    risk_measure = measure
    plt.savefig('Result/{}/{}/{}/{}_har_svr_model{}.png'.format(risk_measure, run_type, cross_validation, run_type, model_type))
    plt.savefig('Result/{}/{}/{}/{}_har_svr_model{}.eps'.format(risk_measure, run_type, cross_validation, run_type, model_type))
'''
calculate the error function between statistics_result['mean'] and RV, 
and save the result by one file in Result folder.
'''


def error_function(y_pred, RV, model_type, measure, run_type, cross_validation):
    import numpy as np
    #RV = RV.shift(-1)[RV.index.isin(statistics_result['mean'].index)]
    y_real = RV.shift(-1)[RV.index.isin(y_pred['mean'].index)].dropna()
    y_pred_1 = y_pred['mean'][y_pred['mean'].index.isin(y_real.index)]

    error = pd.DataFrame()
    error.loc[0, 'RMSE'] = np.sqrt(mean_squared_error(y_real, y_pred_1))
    error.loc[0, 'MAE'] = mean_absolute_error(y_real, y_pred_1)
    error.loc[0, 'MAPE'] = mean_absolute_percentage_error(y_real, y_pred_1)
    error.loc[0, 'R2'] = r2_score(y_real, y_pred_1)
    risk_measure = measure

    error.to_csv('Result/{}/{}/{}/{}_har_svr_error_model{}.csv'.format(risk_measure, run_type, cross_validation, run_type, model_type))


'''
calculate errors between forecast result mean and RV.
'''


def calculate_errors(y_pred, y_true):
    import numpy as np
    error_metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    error = pd.DataFrame(columns=error_metrics)
    error.loc[0] = [np.sqrt(mean_squared_error(y_true, y_pred)),
                    mean_absolute_error(y_true, y_pred),
                    mean_absolute_percentage_error(y_true, y_pred),
                    r2_score(y_true, y_pred)]
    #print('error is {}'.format(error)) 看BUG用的
    return error


'''
calculate the error function between each day forecasting_result and RV,
and calculate mean and variance of error function.
'''


def error_function_each_day(forecasting_result, RV, model_type, run_type, cross_validation, measure):

    # set index to match forecasting_result and RV.

    RV = RV.shift(-1)[RV.index.isin(forecasting_result.index)].dropna()
    forecasting_result = forecasting_result[forecasting_result.index.isin(RV.index)]

    # error label.
    error_metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    errors = pd.DataFrame(columns=error_metrics)
    error_mean = pd.DataFrame(columns=error_metrics)
    error_var = pd.DataFrame(columns=error_metrics)

    # calculate error function of each run in one day.
    for i in range(forecasting_result.columns.size):
        pred_error = calculate_errors(forecasting_result[i], RV)
        #print('pred_error is {},errors is {}'.format(pred_error,errors)) 之前看BUG用的

        if (pred_error.columns != errors.columns).any():
            raise ValueError(''
                             'columns of pred_error is not equal to errors. '
                             'pred_error.columns is {}, errors.columns is {}'.format(pred_error.columns, errors.columns))
        errors.loc[i] = pred_error.loc[0]

    # calculate mean and variance of error function.
    # 计算每个模型的每次运行的误差的均值和方差, 例如运行了10次就是分别计算10个误差，然后再计算这10个误差的平均值和方差

    error_mean.loc[0, 'RMSE'] = errors['RMSE'].mean()
    error_mean.loc[0, 'MAE'] = errors['MAE'].mean()
    error_mean.loc[0, 'MAPE'] = errors['MAPE'].mean()
    error_mean.loc[0, 'R2'] = errors['R2'].mean()

    error_var.loc[0, 'RMSE'] = errors['RMSE'].var()
    error_var.loc[0, 'MAE'] = errors['MAE'].var()
    error_var.loc[0, 'MAPE'] = errors['MAPE'].var()
    error_var.loc[0, 'R2'] = errors['R2'].var()
    risk_measure = measure

    error_mean.to_csv('Result/{}/{}/{}/{}_har_svr_error_mean_model{}.csv'.format(risk_measure, run_type, cross_validation,\
                                                                                 run_type, model_type))
    error_var.to_csv('Result/{}/{}/{}/{}_har_svr_error_var_model{}.csv'.format(risk_measure, run_type, cross_validation,\
                                                                               run_type, model_type))


def timer2(func):
    '''
    calculate the time used for the function which is no return value.
    :param func:
    :return:
    '''

    def wrapper(*args, **kwargs):

        start = time.time()
        result, mcs_result = func(*args, **kwargs)
        end = time.time()
        print('Time used: {} seconds'.format(end - start))
        return result, mcs_result
    return wrapper


@timer2
def main(observation, run_times, num_generations, run_type, cross_validation, other_y=None, other_type=None):
    start = time.time()
    print('HAR model start')
    print('*'*50)

    label_list = ['HAR', 'HAR-SVR1', 'HAR-SVR2', 'HAR-SVR3', 'HAR-SVR4', 'HAR-SVR5', 'HAR-SVR6', 'HAR-SVR7', 'HAR-SVR8', 'HAR-SVR9', 'HAR-SVR10']
    model_type = [2, 3]  # TODO：这里是用来修改输入的model type 数量的
    model_label = label_list[:len(model_type)+1]  # model 的label

    if len(model_label) != len(model_type)+1:
        raise ValueError('model_label length is not equal to model_type length.')

    har_result, use_time = run_har_model(RV=RV, other_y=other_y, other_type=other_type, observation=observation)  # 计算har的结果
    if other_type is None:  # 如果没有输入其他的risk measure，则默认为RV
        risk_measure_har = 'RV'
    else:
        risk_measure_har = other_type

    har_result.to_csv('Result/{}/{}/{}/{}_har_result.csv'.format(risk_measure_har, run_type, cross_validation, run_type))
    print('HAR model end and time used is {} seconds'.format(time.time()-start))
    print('*'*50)
    print('HAR-SVR model start')

    all_result = pd.DataFrame()  # 用来存放在一种risk measure下的所有模型的预测结果的每日的mean，对于har model则是直接存放结果
    all_result = pd.concat([all_result, har_result], axis=1)

    for i in model_type:

        forecasting_result, statistics_result, maximum_val, minimum_val = run_n_times(n=run_times, model_type=i,\
                                                                                      cross_validation=cross_validation,\
                                                                                      observation=observation,\
                                                                                      other_y=other_y, other_type=other_type,\
                                                                                      num_generations=num_generations)
        if other_type is None:
            risk_measure = 'RV'
        else:
            risk_measure = other_type
        forecasting_result.to_csv('Result/{}/{}/{}/{}_har_svr_forecast_result_model{}.csv'.\
                                  format(risk_measure, run_type, cross_validation, run_type, i))
        statistics_result['mean'].to_csv('Result/{}/{}/{}/{}_har_svr_forecast_mean_model{}.csv'.\
                                         format(risk_measure, run_type, cross_validation, run_type, i))
        statistics_result['var'].to_csv('Result/{}/{}/{}/{}_har_svr_forecast_var_model{}.csv'.\
                                        format(risk_measure, run_type, cross_validation, run_type, i))
        maximum_val.to_csv('Result/{}/{}/{}/{}_har_svr_forecast_max_model{}.csv'.\
                           format(risk_measure, run_type, cross_validation, run_type, i))
        minimum_val.to_csv('Result/{}/{}/{}/{}_har_svr_forecast_min_model{}.csv'.\
                           format(risk_measure, run_type, cross_validation, run_type, i))

        error_function(statistics_result, RV, i, measure=risk_measure, run_type=run_type, cross_validation=cross_validation)
        all_result = pd.concat([all_result, statistics_result['mean']], axis=1)
        error_function_each_day(forecasting_result, RV, i, measure=risk_measure, run_type=run_type, \
                                cross_validation=cross_validation)
        plot_result(statistics_result, maximum_val, minimum_val, RV, i, har_result, measure=risk_measure, \
                    run_times=run_times, cross_validation=cross_validation, num_generations=num_generations)


    all_result.columns = model_label
    all_result = all_result.dropna()
    if other_type is None:
        risk_measure = 'RV'
    else:
        risk_measure = other_type
    all_result.to_csv('Result/{}/all_result_for_test_mcs.csv'.format(risk_measure))
    # 计算HAR误差
    calculate_har_error(har_result, RV, measure=risk_measure, run_type=run_type, cross_validation=cross_validation)
    # 计算MCS
    MCS_result = tm.main(all_result, RV)
    print('MCS_result is {}'.format(MCS_result))
    MCS_result.to_csv('Result/{}/{}/{}/{}_MCS_result.csv'.format(risk_measure, run_type, cross_validation, run_type))

    print('Two type HAR-SVR model both end and time used is {} seconds'.format(time.time()-start))
    plt.show()

    return all_result, MCS_result


if __name__ == '__main__':



    measure_list = [None, 'RV+', 'RV-']# SJ的qlike没办法计算，同时harmodel的结果有问题，暂时去掉


    # 这趟运行的是用来干什么的，test代表这趟只是随便跑的测试，0915表示跑的是2009-2015的data
    run_type = '1622'
    all_rm_result = pd.DataFrame()  # 用来把所有的测度下的模型的预测结果都放在一起，然后用MCS来比较不同风险测度下一共12个模型的预测能力
    MCS_result_all = pd.DataFrame()  # 用来把所有的MCS结果放在一起保存


    observationlist = [300, 600, 900, 1200]

    run_times_out = 10
    num_generations_out = 40
    dataset_interval = ['0910', '0915', '1622', '0921']
    data_interval = '1622'
    data_start = 1  # 使用的data的开始点
    data_end = 6  # 使用的data的结束点
    cross_validation_out = 'No split'
    # TODO：添加一个方法来进行不同的observation循环
    for observation in observationlist:
        print(f"Running observation {observation}...")
        observation_folder = f"observation{observation}"

        if not os.path.exists(f"{observation_folder}"):
            os.makedirs(f"{observation_folder}")
            shutil.copytree("Result", f"{observation_folder}/Result")

        current_dir = os.getcwd()
        os.chdir(os.path.join(current_dir, observation_folder))  # 切换到次一级的目录

        for risk_measure in measure_list:
            print('-' * 50)
            start_time = time.time()
            print('Start to run the test program for HAR-SVR model.')
            print('-' * 50)
            print('Loading data...')

            # ------实际运行时候的指令
            all_data = bf.getdata(interval=data_interval, year_start=data_start, year_end=data_end)
            all_data = bf.concatRV(all_data)
            if risk_measure is None:
                RV = bf.calculRV(all_data, interval=data_interval)
                other_y = None
            else:
                other_y, RV = bf.calculRV(all_data, interval=data_interval, type=risk_measure)
                other_y.index = pd.to_datetime(other_y.index, format='%Y%m%d')

            '''根据不同的observation，统一后面的测试集的时间区间'''
            RV = RV.iloc[max(observationlist) - observation:, :]

            # ------实际运行时候的指令
            # ----------------test用的指令----------------
            # RV = pd.read_csv('data/test_data/test_data.csv', index_col=0)  # data数：100
            # ----------------test用的指令----------------
            RV.index = pd.to_datetime(RV.index, format='%Y%m%d')


        # TODO:增加concat data的功能，使其能够计算SJ，RV+，RV-，RV
        # 功能加完了，就是不知道对不对，测试也通过了

            if observation > len(RV):
                raise ValueError('The observation is larger than the length of RV.')

            print(f'Data loaded. Loading data used {time.time() - start_time} seconds.')
            print('-' * 50)

            result, mcs_result = main(other_y=other_y, other_type=risk_measure, observation=observation, run_times=run_times_out, \
                 num_generations=num_generations_out, cross_validation=cross_validation_out, run_type=run_type)

            all_rm_result = pd.concat([all_rm_result, result], axis=1)
            MCS_result_all = pd.concat([MCS_result_all, mcs_result], axis=1)


        all_rm_result.to_csv('Result/allresult/all_rm_result.csv')
        MCS_result_all.to_csv('Result/allresult/MCS_result_all.csv')

        mcs_result_all = tm.main(all_rm_result, RV)
        print('mcs_result_all is {}'.format(mcs_result_all))

        mcs_result_all.to_csv('Result/allresult/mcs_result_all.csv')


'''

'''




