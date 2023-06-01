import basicFunction as bf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    """
    load data from database
    :return: high-frequency data
    """
    data = bf.getdata(interval='1622', year_start=1, year_end=6, system='mac')
    data = bf.concatRV(data)

    return data


def calculate_risk_measure(risk_measure):

    """
    calculate the risk measure
    :param risk_measure:
    :return:
    """

    data = load_data()
    if risk_measure == 'RV':
        measure_data = bf.calculRV(data, interval='1622')
        measure_data.index = pd.to_datetime(measure_data.index, format='%Y%m%d')
        return measure_data

    if risk_measure != 'RV':

        measure_data, _ = bf.calculRV(data, interval='1622', type=risk_measure)
        measure_data.index = pd.to_datetime(measure_data.index, format='%Y%m%d')

        return measure_data


def plot_data(data, risk_measure):

    sns.set()
    data.plot(figsize=(16, 9))
    ylabel_set = {'RV': 'Realized Volatility',
                  'RV+': 'Realized Volatility +',
                  'RV-': 'Realized Volatility -',
                  'SJ': 'Signed Jumps',
                  'SJ_abs': 'Absolute Value SJ'}

    plt.legend([f'{risk_measure}'])
    plt.xlabel('Date')
    plt.ylabel(f'{ylabel_set[risk_measure]}')
    plt.title(f'{ylabel_set[risk_measure]} Data')
    plt.show()

    plt.savefig(f'/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/risk_measure_figure/{risk_measure}_data.eps', dpi=600)
    plt.savefig(f'/Users/zhuoyue/Documents/PycharmProjects/HAR_RV/risk_measure_figure/{risk_measure}_data.png', dpi=600)


def main():

    risk_measure_list = ['RV', 'RV+', 'RV-', 'SJ', 'SJ_abs']

    for risk_measure in risk_measure_list:

        data = calculate_risk_measure(risk_measure)
        plot_data(data, risk_measure)


if __name__ == '__main__':

    main()
