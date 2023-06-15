import os
import pandas as pd
import test_MCS as tm
import basicFunction as bf


def load_y_pred(measure_name):
    """
    load each measures data
    :return:  data
    """
    print('loading y_pred data ...')
    os.chdir('/Users/zhuoyue/Downloads/result/observation1200/Result/allresult')
    try:
        data = pd.read_csv(f'mcs_result_each_measure_{measure_name}.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
    except FileNotFoundError:
        return None
    return data


def load_y():
    """
    load the real data
    :return:
    """
    print('loading y data ...')
    data_interval = '1622'
    system_name = 'mac'
    data_start = 1  # 使用的data的开始点
    data_end = 6
    all_data = bf.getdata(interval=data_interval, year_start=data_start, year_end=data_end, system=system_name)
    all_data = bf.concatRV(all_data)

    RV = bf.calculRV(all_data, interval=data_interval)
    RV.index = pd.to_datetime(RV.index, format='%Y%m%d')

    return RV


def calculate_mcs(y_pred, y):
    """
    calculate the MCS
    :param data:
    :return: MCS p-value
    """
    print('calculating MCS ...')
    mcs_result = tm.main(y_pred, y)

    return mcs_result


if __name__ == '__main__':

    measure_name_list = ['RV', 'RV+', 'RV-', 'SJ', 'SJ_abs']
    for measure_name in measure_name_list:
        y_pred = load_y_pred(measure_name)
        if y_pred is None:
            continue
        y = load_y()
        forecast, y_, forecast_columns = tm.match_data(y_pred, y)
        mcs_result = calculate_mcs(y_pred, y)
        mcs_result.to_csv(f'/Users/zhuoyue/Downloads/result/observation1200/Result/mcs_result_for_each_measure_{measure_name}.csv')
