from har_svrmodel import HARSVRmodel
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)


class HARsvrResult(HARSVRmodel):

    def __init__(self, y, har_model, ga_svr_model, alpha = None, *args, **kwargs):

        super().__init__(y, *args, **kwargs)
        self.har_model = har_model
        self.ga_svr_model = ga_svr_model
        self.har_ga_svr_resid = self.get_hargasvr_resid()
        self.har_resid = self.get_har_resid()
        self.r2square = self.get_hargasvr_r2
        self.mape = self.get_hargasvr_mape
        self.mse = self.get_hargasvr_mse
        self.mae = self.get_hargasvr_mae
        self.alpha = alpha

    def get_hargasvr_resid(self):
        if self.model_type ==1:
            actual_x, actual_rv = self.get_SVR_data_train()
            har_ga_svr_resid = actual_rv - self.har_model().predict(actual_x)+\
                self.ga_svr_model.predict(self.HAR_model(y=self.y, lags = self.lags).regressor)
        elif self.model_type ==2:
            alpha = self.alpha
            actual_x, actual_rv = self.get_SVR_data_train()
            har_ga_svr_resid = actual_rv - (1 - alpha) * self.ga_svr_model.predict(actual_x) + \
                     alpha * self.har_model.predict(actual_x)

        return har_ga_svr_resid

    def get_har_resid(self):

        har_resid = self.har_model.resid

        return har_resid

    def predict(self, step=1):
        '''
        :param step :  Number of steps for forecasting
        :param predict_data: 用来作预测的data
        :return: 预测出来的结果
        '''
        if self.model_type == 1:
            predict_data = self.har_model.out_resid
        elif self.model_type ==2:
            predict_data = self.har_model.out_predictor

        x = self.get_predict_data(predict_data, days=step)
        if self.model_type == 1:
            result = self.ga_svr_model.predict(x) + self.har_model.out_result
        elif self.model_type == 2:
            alpha = self.get_alpha()
            result = (1 - alpha) * self.ga_svr_model.predict(x) + \
                     alpha * self.har_model.out_result

        return result

    def properties_calculator(func):
        '''
        用来计算样本内的预测值从而计算各种统计量
        :param func: 传入接下来要用的function
        :return: 统计量
        '''

        def wrapper(self):
            actual_x, actual_rv = self.get_SVR_data_train()
            har_model = self.har_model
            ga_svr_model = self.ga_svr_model
            if self.model_type == 1:
                predict_y = har_model.predict(actual_x) + \
                            ga_svr_model.predict(actual_x)
            elif self.model_type == 2:
                alpha = self.get_alpha()
                predict_y = (1 - alpha) * ga_svr_model.predict(actual_x) + \
                            alpha * har_model.predict(actual_x)

            result = func(actual_rv, predict_y)

            return result
        return wrapper

    @properties_calculator
    def get_hargasvr_r2(self, actual_rv, predict_y):

        return r2_score(actual_rv, predict_y)

    @properties_calculator
    def get_hargasvr_mse(self, actual_rv, predict_y):

        return mean_squared_error(actual_rv, predict_y)

    @properties_calculator
    def get_hargasvr_mae(self, actual_rv, predict_y):

        return mean_absolute_error(actual_rv, predict_y)

    @properties_calculator
    def get_hargasvr_mape(self, actual_rv, predict_y):

        return mean_absolute_percentage_error(actual_rv, predict_y)









