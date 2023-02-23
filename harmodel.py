import pandas as pd
from statsmodels.api import OLS

class HARX:
    '''
    reference : CORSI (2009)，Audrino,Huang,Okhrin(2018)

    estimation method :OLS and LASSO（Flexible HAR Model）

    estimate by

    y_t = c + sum(i=0,p)RV_i + gamma*x + epsilon_t

    c: constant.
    RV_i : lags days RV
    x : exogenous variable.
    epsilon_t : error at time t.

    har_svr的所有X相关的回归参数都是和regreesor相关的，如果增加几个测度来预测，修改regreesor的生成函数就可以了。
    in_target不需要修改，因为都是用RV的。
    '''
    #TODO：添加一个传入参数，用来接收其他种类的measure，比如SJ，RV+，RV-，RV
    def __init__(self, y, other_type=None,other_y=None, x=None, z=None,
                 lags=[1, 5, 22],
                 Method='OLS',
                 observation=300,
                 step=1):
        '''
        :param other_y: 当需要用作自变量的是 RV+，RV-，SJ的时候用来传入这些数据
        :param other_type: RV+，RV-，SJ
        :param z: 当传入的y是处理好的多个RV的时候用来传入训练的target
        :param y: dependent variable
        :param x: exogenous regressors default ： None.
        :param lags: Description of lag structure of the HAR.
        :param Method: estimation method . default : OLS
        :param observation: 需要用多少天的数据量来建立数据
        :param step: Number of steps to forecast.
        '''
        other_type_set = ['RV+', 'RV-', 'SJ', None]
        if other_type not in other_type_set:
            raise ValueError('other_type must be in {}'.format(other_type_set))

        self.other_y = other_y
        self.other_type = other_type
        self.z = z
        self.y = y
        self.lags = lags
        self.x = x
        self.Method = Method
        self.observation_num = observation
        self.step = step

        self.fit_start,self.fit_stop = self.fit_start_end()
        if len(y.columns) < len(self.lags):
            self.regressor = self.regressor_generator()[:self.observation_num].dropna()
        else:
            self.regressor = self.y
        self.in_target = self.in_target_generator()
        self.predictor = self.predictor_generator()
        self.out_target = self.out_target_generator()
        self.out_predictor = self.out_predictor_generator()


    def regressor_generator(self):
        '''

        :return: 用作independet variable 的参数
        '''
        if self.other_y is None:
            RVdata = pd.DataFrame()
            y = self.y
            lags = self.lags

            for i in range(len(self.lags)):

                RVdata['{} day RV'.format(lags[i])] = y.rolling(window=lags[i]).mean()

            return RVdata
        elif self.other_y is not None:
            result = pd.DataFrame()
            y = self.other_y
            lags = self.lags

            for i in range(len(self.lags)):
                result['{} day {}'.format(lags[i], self.other_type)] = y.rolling(window=lags[i]).mean()

            return result

    def fit_start_end(self):
        '''
        用来找到数据做完了之后使用的是第几个数据开始训练，以及第几个参数处结束。
        :return:fit_start and fit_end
        '''
        fit_start = max(self.lags)-1
        fit_end = self.observation_num

        return fit_start,fit_end

    def in_target_generator(self):
        '''
        generate in-sample dependent variable for HAR model.
        :return: in-sample dependent variable.
        '''

        observation = self.observation_num
        # 无论用什么进行回归，都是用传进来的y当作因变量
        if len(self.y.columns) < len(self.lags):
            in_target = self.y.shift(-1)[:observation]
        else:
            in_target = self.z.shift(-1)  # 当传进来的是进行过多日平均计算，并且已经排列好了的自变量的时候。

        return in_target

    def predictor_generator(self):
        '''

        generate out-sample independent variable data for HAR model.
        这的data是区分出来的样本外data。
        len(self.y.columns) <2用来判断传进来的数据是处理好了的多天的RV还是没有处理过的单日RV
        :return: out-sample independent variable.

        '''
        # regressor 的地方判断好了就行。当使用排列好的other_y进行回归的时候，other_y就会变成y
        if len(self.y.columns) < len(self.lags):
            predictor = self.regressor_generator()[self.observation_num:]
        else:
            predictor = self.y[self.observation_num:]  # 如果传进来的是排列好的y的时候，好像有问题。
        return predictor

    def out_predictor_generator(self):
        '''
        这是用来根据想要预测的step来从样本外data选择data
        :return: out-sample independent variable.
        '''

        step = self.step
        out_predictor = self.predictor[:step]

        return out_predictor

    def out_target_generator(self):
        '''

        :return: out-sample target for to calculate out-sample residuals.
        '''

        observation = self.observation_num
        out_target = self.y[observation:observation+self.step]

        return out_target

    def in_out_result(self, model):
        '''
        :param model: 要用来求in-sample和out-sample预测值的model
        用来获取in-sample和out-sample的预测值
        :return: in-sample and out-sample
        '''

    def fit(self):

        '''
        #TODO:所有len(self.y.columns) < len(self.lags)的地方都是因为没有完善直接传入处理好了的RV data的缘故。
        :param self.out_resid : out-sample residuals.
        :return: fitted model by estimation method.

        '''
        regressor = self.regressor
        if len(self.y.columns) <len(self.lags):
            target = self.in_target[self.fit_start:]
        else:
            target = self.in_target


        if self.Method == 'OLS':
            self.HAR_MODEL = OLS(endog=target, exog=regressor, missing='drop').fit()

        if len(self.y.columns) < len(self.lags):
            self.in_result = self.HAR_MODEL.predict(exog=regressor).to_frame()
            self.in_result.columns = self.in_target.columns
            self.out_result = self.HAR_MODEL.predict(self.out_predictor).to_frame()
            self.out_result.columns = self.out_target.columns
            self.out_resid = self.out_target - self.out_result

        return self.HAR_MODEL


