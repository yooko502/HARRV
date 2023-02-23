from harmodel import HARX
from sklearn.svm import SVR
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import TimeSeriesSplit
import pygad


class HARSVRmodel():

    '''
    reference : CORSI (2009)，Audrino,Huang,Okhrin(2018)

    estimation method :OLS and LASSO（Flexible HAR Model）

    HAR model : estimated by arch.univariate

    OLS method : 使用sklearn 的LinearRegression来实现
    type 1:
    HAR+SVR model: SVR for forecasting residual.
    HAR model : by own.
    SVR model : sklearn package.
    type 2:
    HAR type SVR model: x: RV[1,5,22]
    RV = alpha*RV_HAR +(1-alpha)*RV_SVR
    alpha decided by GA.
    需要优化的参数比type 1 多一个alpha

    :return:HAR-GA-SVR MODEL
    '''

    def __init__(self, y, x = None,
                 lags = [1,5,22], observation=100,##HAR model parameter
                 modeltype = 2, # 需要使用的模型类型
                 num_generations=200, num_parents_mating=16,  ##GA-SVR需要的参数
                 sol_per_pop=25, num_genes=3,
                 gene_space=[{'low': 1e-6, 'high': 100}, {'low': 1e-2, 'high': 200}, {'low': 1e-6, 'high': 100}],
                 gene_type=[float, 6],
                 crossover_type='two_points',
                 crossover_probability=0.6,
                 mutation_type='random',
                 mutation_probability=0.6,
                 last_fitness=0,
                 fitnessFunction='MAE',
                 Methodkeys='No split',
                 typeOffitness={},
                 keep_parents=10,
                 initial_population=None,
                 if_print=True,
                 sliding_window_size=3,
                 plot_fig=False,
                 useData='train',
                 testDatalengthRate=1/2
                 ):
        '''

        :param num_genes: 设定有多少个基因（即需要优化的参数有几个）
        :param gene_space: 基因的取值范围
        :param gene_type:基因的类型，例如int float etc... 第二个是保留的小数点位数
        :param crossover_type: 交叉方法 GA的crossover方法
        :param crossover_probability: crossover 的机率
        :param mutation_type: 如何进行基因的变异
        :param mutation_probability: 变异的机率
        :param last_fitness: 最后一项的fitness值
        :param fitnessFunction: 用来进行优化的时候使用的fitnessFunction类型 MAE r2 etc...
        :param Methodkeys: 是否进行交叉验证
        :param typeOffitness:
        :param initial_population: 需要优化的参数的初始值
        :param if_print:x
        :param sliding_window_size: x
        :param plot_fig:x
        :param useData: 用样本内结果还是样本外结果来优化SVR
        :param testDatalengthRate:x
        :param x: 调整好的用作independent variable的data
        :param y: 调整好的用作dependent variable的data
        :param observation: 用作训练的数据的大小 默认 100天
        :param lags :HAR model的Realized volatility的lag.默认[1,5,22]
        :param num_generations : GA 跑多少个世代
        :param num_parents_mating :选多少个当作下一世代的parents
        :param keep_parents :在下个世代保留多少个parents
        :param sol_per_pop :每一个世代有多少个备选

        '''

        self.model_type = modeltype

        self.x = x
        self.y = y
        self.lags = lags
        self.observation = observation

        self.fitnessFunction = fitnessFunction
        self.Methodkeys = Methodkeys
        self.typeOfitness = typeOffitness
        # GAsetting
        # ga参数设定
        self.gene_type = gene_type
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.keep_parents = keep_parents
        self.sol_per_pop = sol_per_pop
        self.num_genes = self.num_genes_jug(modeltype,num_genes)
        self.gene_space = self.modeltype_jug(modeltype, gene_space)
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.last_fitness = last_fitness
        self.initial_population = initial_population
        self.if_print = if_print
        self.sliding_window_size = sliding_window_size
        self.useData = useData
        self.plot_fig = plot_fig

    '''
    
    HAR_model_fit : 
    
    '''
    def modeltype_jug(self,modeltype,gene_space):

        '''

        :param modeltype: model type .1 or 2
        :param gene_space: gene_space for GA.
        :return: gene_space for different type model.
        '''

        if modeltype == 1 :
            return gene_space
        elif modeltype == 2 :
            gene_space.append({'low': 0, 'high': 1})
            return gene_space

    def num_genes_jug(self,modeltype,num_gene):
        '''
        model 1 or 2 对应不同的基因数量
        :param modeltype: model type .1 or 2
        :param num_gene: gene's number
        :return: model 1 or 2 's number
        '''

        if modeltype == 1 :
            return num_gene
        elif modeltype == 2 :
            return num_gene+1

    def HAR_model_fit(self):

        '''

        :return: fitted HARX model by arch.univariate
        '''

        harx = HARX(y=self.y, lags=self.lags)
        harx = harx.fit()

        return harx

    def HAR_model(self):

        '''
        使用想使用的前observation个观测值进行建立模型
        :return: HARX model
        '''

        harx = HARX(y=self.y, lags = self.lags)

        return harx

    def getresid(self):

        '''

        用来获取训练过后的HAR model的残差

        :return: HAR model's residual.

        '''

        residual = self.HAR_model_fit().resid

        return residual

    def svr_x_estimator(self):
        '''


        :return: type-1 model's x.

        '''

        pass

    def getSVRdata(self):
        '''
        用来获取给SVR用的X和Y

        type-2 : HAR-type SVR model .
        type-1 : SVR model for forecasting error.

        :return: X, Y
        '''

        if self.model_type == 1 :
            # TODO：这边的X还没有选择好，参考GARCH-SVR那篇文章的选择方法

            Y = self.getresid()[self.HAR_model().fit_start:]
            X = 1
            return X, Y

        elif self.model_type == 2 :

            #TODO:SVR的model2的训练集暂时没问题
            Y = self.HAR_model().in_target[self.HAR_model().fit_start:]
            X = self.HAR_model().regressor

            return X, Y

    def getpredictdata(self, data, days = 1):
        '''

        :param data: 用来作预测的data
        :param days: 要预测将来多少天 默认1
        :return: data的切片
        '''
        independentV = data[:days]

        return independentV

    def SVRmodel(self,gamma = 1 , epsilon = 0.01, C = 1, kernaltype = 'rbf'):
        '''

        :param x: predictor. A matrix or a dataframe.
        :param y: target variable. Vector or a column of values
        :param gamma: SVR's parameter.  width of the kernel function's influence
        :param epsilon: SVR's parameter. Tolerance of error
        :param C: SVR's parameter. Penalty factor
        :param kernaltype:type of kernal function . Like 'rbf', 'linear', etc...
        :return: SVR_model
        '''

        SVR_model = SVR(kernel=kernaltype, gamma = gamma,C = C, epsilon = epsilon)

        return SVR_model

    def weightHARSVR(self):

        pass

    def ga_svr(self, testDatalengthRate=5):

        '''

        :param testDatalengthRate: 用来对训练集和测试集做区分的
        :return:  设定好参数的GA-SVRmodel
        '''

        # TODO： 训练集和测试集的区分方法还有待优化
        self.train_X,self.train_y = self.getSVRdata()

        if self.useData =='test':
            self.train_X,self.train_y = self.train_X[:1-len(self.train_X)//testDatalengthRate],\
                                        self.train_y[:1-len(self.train_y)//testDatalengthRate]
            self.test_X,self.test_y = self.train_X[1-len(self.train_X)//testDatalengthRate:],\
                                      self.train_y[1-len(self.train_y)//testDatalengthRate:]

        def on_generation(ga_instance):
            if self.if_print == True:

                print("")
                print("Generation = {generation}".format(generation=ga_instance.generations_completed))
                print("Fitness    = {fitness}".format(
                    fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
                print("Change     = {change}".format(
                    change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[
                               1] - self.last_fitness))
                self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        def fitness_function(solution,solution_idx):
            # fitnesstype = fitnessFunc, splittype = SplitMethod
            '''
            使用验证集数据来进行GA-SVR的参数优化

            model-type 2 : 只支持不使用验证集的计算。即 useData == train

            '''
            if self.Methodkeys == 'No split':
                if self.useData == 'test':

                    svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])

                    svr_prediction = svr_optimization.fit(X=self.train_X, y=self.train_y).predict(self.test_X)
                    # print('svr_prediction is {}'.format(svr_prediction))
                    i = int(len(self.test_X)*testDatalengthRate)
                    if self.fitnessFunction == 'MAE':
                        # fitness = 0
                        # for i in range(1,int(len(self.test_X)/2)):
                        fitness = mean_absolute_error(self.test_y[:i], svr_prediction[:i])  # MAE
                        return -fitness
                    elif self.fitnessFunction == 'r2':
                        # fitness = 0
                        # for i in range(1,int(len(self.test_X)/2)):
                        fitness = r2_score(self.test_y[:i], svr_prediction[:i])  # r2
                        return fitness
                    elif self.fitnessFunction == 'MAPE':
                        # fitness = 0
                        # for i in range(1,int(len(self.test_X)/2)):
                        fitness = mean_absolute_percentage_error(self.test_y[:i], svr_prediction[:i])  # r2
                        return -fitness
                    elif self.fitnessFunction == 'MSE':
                        # fitness = 0
                        # for i in range(1,int(len(self.test_X)/2)):
                        fitness = mean_squared_error(self.test_y[:i], svr_prediction[:i])  # r2
                        return -fitness

                elif self.useData == 'train':
                    if self.model_type == 1 :
                        svr_optimization = SVR(kernel= 'rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])

                        svr_prediction = svr_optimization.fit(X=self.train_X, y=self.train_y).predict(self.train_X)
                        # print('svr_prediction is {}'.format(svr_prediction))
                        i = int(len(self.test_X) * 2 / 3)
                        if self.fitnessFunction == 'MAE':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = mean_absolute_error(self.train_y, svr_prediction)  # MAE
                            return -fitness
                        elif self.fitnessFunction == 'r2':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = r2_score(self.train_y, svr_prediction)  # r2
                            return fitness
                        elif self.fitnessFunction == 'MAPE':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = mean_absolute_percentage_error(self.train_y, svr_prediction)  # r2
                            return -fitness
                        elif self.fitnessFunction == 'MSE':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = mean_squared_error(self.train_y, svr_prediction)  # r2
                            return -fitness
                    #TODO:完善对于使用验证集data的支持。
                    elif self.model_type == 2 :

                        svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])
                        svr_prediction = (1-solution[3])*svr_optimization.fit(X=self.train_X,\
                                                                              y=self.train_y).predict(self.train_X)\
                                        +solution[3]*self.HAR_model_fit().predict(self.train_X)

                        if self.fitnessFunction == 'MAE':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = mean_absolute_error(self.train_y, svr_prediction)  # MAE
                            return -fitness
                        elif self.fitnessFunction == 'r2':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = r2_score(self.train_y, svr_prediction)  # r2
                            return fitness
                        elif self.fitnessFunction == 'MAPE':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = mean_absolute_percentage_error(self.train_y, svr_prediction)  # MAPE
                            return -fitness
                        elif self.fitnessFunction == 'MSE':
                            # fitness = 0
                            # for i in range(1,int(len(self.test_X)/2)):
                            fitness = mean_squared_error(self.train_y, svr_prediction)  # MSE
                            return -fitness

            elif self.Methodkeys == 'TSCV':

                svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])
                tscv = TimeSeriesSplit()

                ScoreOfmodel = 0
                count = 0
                for train_index, test_index in tscv.split(self.train_X):

                    # print("TRAIN:", train_index, "TEST:", test_index)

                    X_train, X_test = self.train_X[train_index], self.test_X[test_index]

                    y_train, y_test = self.train_y[train_index], self.test_y[test_index]

                    svr_prediction = svr_optimization.fit(X=X_train, y=y_train).predict(X_test)
                    # print('svr_prediction is {}'.format(svr_prediction))
                    if self.fitnessFunction == 'MAE':
                        ScoreOfmodel = ScoreOfmodel + mean_absolute_error(y_test, svr_prediction)  # MAE

                    # print('MAE score is :', mean_absolute_error(y_test, svr_prediction))

                    elif self.fitnessFunction == 'r2':
                        ScoreOfmodel = ScoreOfmodel + r2_score(y_test, svr_prediction)  # r2
                    # print('r2 score is :', mean_absolute_error(y_test, svr_prediction))

                    elif self.fitnessFunction == 'MSE':
                        ScoreOfmodel = ScoreOfmodel + mean_squared_error(y_test, svr_prediction)

                    elif self.fitnessFunction == 'MAPE':
                        ScoreOfmodel = ScoreOfmodel + mean_absolute_percentage_error(y_test, svr_prediction)

                    count += 1

                fitness = ScoreOfmodel / count

                if self.fitnessFunction == 'MAE':
                    if self.if_print == True:
                        print("fitness is {}".format(-fitness))

                    return -fitness
                elif self.fitnessFunction == 'r2':
                    if self.if_print == True:
                        print("fitness is {}".format(fitness))
                    return fitness

                elif self.fitnessFunction == 'MSE':
                    if self.if_print == True:
                        print("fitness is {}".format(-fitness))
                    return -fitness

                elif self.fitnessFunction == 'MAPE':
                    if self.if_print == True:
                        print("fitness is {}".format(-fitness))
                    return -fitness

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=self.num_genes,
                               gene_space=self.gene_space,
                               on_generation=on_generation,
                               gene_type=self.gene_type,
                               # save_solutions= True,
                               # save_best_solutions= True,
                               mutation_type=self.mutation_type,
                               mutation_probability=self.mutation_probability,
                               crossover_type=self.crossover_type,
                               crossover_probability=self.crossover_probability,
                               initial_population=self.initial_population
                               )

        ga_instance.run()



        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        self.solution_fitness = solution_fitness  # 方便在外面存取最佳的fitness
        self.solution = solution  # 存取最佳的solution
        print("Parameters of the best solution : gamma = {},C = {}, epsilon = {}".format(solution[0], solution[1],
                                                                                         solution[2]))
        if self.model_type == 2:
            print('best weight is {}'.format(solution[3]))

        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        print('The fitness function is {}'.format(self.fitnessFunction))

        gasvr = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])

        return gasvr

    def get_alpha(self):
        '''
        only for model-type 2.
        result = alpha*HAR + (a-alpha)*SVR
        :return: alpha
        '''
        alpha = self.solution[3]
        return alpha

    def fit(self):

        X, Y = self.getSVRdata()
        gasvr = self.ga_svr()

        gasvr_fit = gasvr.fit(X=X, y=Y)
        har_model = self.HAR_model_fit()
        if self.model_type == 2:
            alpha = self.get_alpha()

            return HARsvrResult(y=self.y, alpha=alpha, ga_svr_model=gasvr_fit, har_model=har_model)
        elif self.model_type == 1:

            return HARsvrResult(y=self.y, alpha=None, ga_svr_model=gasvr_fit, har_model=har_model)


class HARsvrResult(HARSVRmodel):

    def __init__(self, y, har_model, ga_svr_model, alpha, *args, **kwargs):

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
            actual_x, actual_rv = self.getSVRdata()
            har_ga_svr_resid = actual_rv - self.har_model().predict(actual_x)+\
                self.ga_svr_model.predict(self.HAR_model(y=self.y, lags = self.lags).regressor)
        elif self.model_type ==2:
            alpha = self.alpha
            actual_x, actual_rv = self.getSVRdata()
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

        x = self.getpredictdata(predict_data, days=step)
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


















