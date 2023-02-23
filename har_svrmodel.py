from harmodel import HARX
from sklearn.svm import SVR
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import TimeSeriesSplit
import pygad
import numpy as np
import pandas as pd




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

    def __init__(self, y, other_type=None, other_y=None, x=None,
                 lags=[1,5,22], observation=300,  # HAR model parameter
                 modeltype=2,  # 需要使用的模型类型
                 num_generations=2000, num_parents_mating=16,  # GA-SVR需要的参数
                 sol_per_pop=25, num_genes=3,
                 gene_space=[{'low': 1e-5, 'high': 100}, {'low': 1e-6, 'high': 200}, {'low': 1e-5, 'high': 100}],#从左到右，gamma C epsilon
                 gene_type=[float, 6],
                 crossover_type='two_points',
                 crossover_probability=0.8,
                 mutation_type='random',
                 mutation_probability=0.8,
                 last_fitness=0,
                 fitnessFunction='MAE',
                 Methodkeys='No split',
                 typeOffitness={},
                 keep_parents=15,
                 initial_population=None,
                 if_print=True,
                 sliding_window_size=3,
                 plot_fig=False,
                 useData='test',  # 用train的话 model1 会报错，但是暂时不想改
                 testDatalengthRate=1/2
                 ):
        '''
        :param other_y: 当需要用作自变量的是 RV+，RV-，SJ的时候用来传入这些数据
        :param other_type: RV+，RV-，SJ
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
        other_type_set = ['RV+', 'RV-', 'SJ', None]
        if other_type not in other_type_set:
            raise ValueError('other_type must be in {}'.format(other_type_set))

        self.other_y = other_y
        self.other_type = other_type

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
        self.num_genes = self.num_genes_jug(num_genes)
        self.gene_space = self.modeltype_jug(gene_space)
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
    def modeltype_jug(self,gene_space):

        '''

        :param modeltype: model type .1 or 2
        :param gene_space: gene_space for GA.
        :return: gene_space for different type model.
        '''
        # TODO: 2/20 23:54新增了 model type 3 还没有测试。区别是比model 1 多了一个特征，（RV_d）^2。
        if self.model_type == 1 or self.model_type == 3:
            return gene_space
        elif self.model_type == 2 :
            gene_space.append({'low': 0, 'high': 1})
            return gene_space

    def num_genes_jug(self, num_gene):
        '''
        model 1 or 2 对应不同的基因数量
        :param model_type: model type .1 or 2
        :param num_gene: gene's number
        :return: model 1 or 2 's number
        '''

        if self.model_type == 1 or self.model_type == 3:
            return num_gene
        elif self.model_type == 2 :
            return num_gene+1

    def HAR_model_fit(self,y=None, x=None):

        '''

        :return: fitted HARX model by arch.univariate
        '''
        # TODO:还没有增加RV+RV-SJ的判断
        if y is not None:   # step 1:判断传进来的是不是排列好的数据，排列好的话，那无论如何都可以直接进行回归。
            harx = HARX(y=x, z=y, lags=self.lags, observation=self.observation)
            harx = harx.fit()
        else:
            if self.other_y is None:  # step 2:判断是否有other_y，如果没有，那就直接进行回归。
                harx = HARX(y=self.y, lags=self.lags, observation=self.observation)
                harx = harx.fit()
            elif self.other_y is not None:
                harx = HARX(y=self.y, other_y=self.other_y, other_type=self.other_type, lags=self.lags,\
                            observation=self.observation)
                harx = harx.fit()

        return harx

    def HAR_model(self,observation=None):

        '''
        使用想使用的前observation个观测值进行建立模型
        :return: HARX model
        '''

        if self.other_y is None:

            if observation is None:
                harx = HARX(y=self.y, lags=self.lags, observation=self.observation)
            else:
                harx = HARX(y=self.y, lags=self.lags, observation=observation)

        elif self.other_y is not None:

            if observation is None:
                harx = HARX(y=self.y, other_y=self.other_y, other_type=self.other_type, lags=self.lags, observation=self.observation)
            else:
                harx = HARX(y=self.y, other_y=self.other_y, other_type=self.other_type, lags=self.lags, observation=observation)

        return harx

    def get_resid(self):

        '''

        用来获取训练过后的HAR model的in-samle残差

        :return: HAR model's residual.

        '''

        residual = self.HAR_model_fit().resid

        return residual

    def svr_x_generator(self):
        '''
        generate x for SVR model.

        :return: type-1 model's x.

        '''
        X = self.HAR_model().regressor
        # TODO:resid 要shift(1),不然拼起来的不对。
        resid = pd.DataFrame(self.get_resid(),index=X.index).shift(1).dropna()
        result = pd.merge(X,resid,on='date').dropna()
        if self.model_type == 3:
            result = pd.merge(result, X.loc[:, X.columns[0]] ** 2, on='date').dropna()
        return result

    def get_SVR_data_train(self):
        '''
        用来获取给SVR用的X和Y for training

        type-2 : HAR-type SVR model .-> alpha*RV_HAR + (1-alpha)*RV_SVR
        type-1 : SVR model for forecasting error. -> RV_HAR + error_SVR

        :return: X, Y
        '''

        if self.model_type == 1 or self.model_type == 3:
            # 原本Y后面是shift（-1）这里和svr_x_generator只要有一遍shift就行了
            Y = self.get_resid().dropna()
            X = self.svr_x_generator()
            # Find the common index
            # Remove the rows with index values that are not common
            if len(X.index) > len(Y.index):
                X = X[X.index.isin(Y.index)]
            else:
                Y = Y[Y.index.isin(X.index)]
            X.columns = X.columns.astype(str)
            Y.columns = ['resid data']
            if len(X) != len(Y):
                raise ValueError('SVR training set X and Y have different length')
            return X, Y

        elif self.model_type == 2 :

            #SVR的model2的训练集如果不取和Y一样的index则会出现空值
            Y = self.HAR_model().in_target[self.HAR_model().fit_start:].dropna()
            #X= self.HAR_model().regressor
            X = self.HAR_model().regressor[self.HAR_model().regressor.index.isin(Y.index)]
            return X, Y

    def get_SVR_data_test(self):
        '''
        only for type 1 model.
        get X of fitted SVR model for testing.
        :return: X
        '''
        #TODO：等到确定type1的model的SVR的自变量形式之后确定X

        x = self.svr_x_generator().dropna(axis=0).tail(1)
        return x

    def get_predict_data(self, data, days=1):
        '''

        :param data: 用来作预测的data
        :param days: 要预测将来多少天 默认1
        :return: data的切片
        '''
        independentV = data[:days]

        return independentV

    def SVRmodel(self,gamma=1, epsilon=0.01, C=1, kernaltype='rbf'):
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

    def ga_svr(self, testDatalengthRate=5):

        '''

        :param testDatalengthRate: 用来对训练集和测试集做区分的
        :return:  设定好参数的GA-SVRmodel
        '''

        # TODO： 训练集和测试集的区分方法还有待优化
        # 根据选择的训练集不同来获取不同的训练集和测试集
        self.train_X,self.train_y = self.get_SVR_data_train()

        train_X, train_y = self.get_SVR_data_train()

        # 表明使用test的数据来进行fitness的反馈
        if self.useData == 'test':

            if self.Methodkeys == 'No split':

                split_idx = len(train_X) - len(train_X) // testDatalengthRate
                self.train_X, self.train_y = train_X[:split_idx], train_y[:split_idx]
                self.test_X, self.test_y = train_X[split_idx:], train_y[split_idx:]
                #TODO:需改了一下if 语句的位置，还没有测试。2/16 15:05, 2/16 15:21 第二次修改
                if self.model_type == 1 or self.model_type == 3:
                    in_target_HAR = self.HAR_model().in_target[self.HAR_model().in_target.index.isin(train_y.index)]
                    regressor_HAR = self.HAR_model().regressor[self.HAR_model().regressor.index.isin(train_X.index)]

                    HAR_train_X, HAR_train_y = regressor_HAR[:split_idx], in_target_HAR[:split_idx]
                    HAR_test_X, HAR_test_y = regressor_HAR[split_idx:], in_target_HAR[split_idx:]
                    if len(HAR_test_X) != len(HAR_test_y):
                        raise ValueError('HAR_test_X and HAR_test_y must have the same length,\n'
                                         'in_target_HAR length is {},and regressor_HAR length is {},\n'
                                         'HAR_test_X length is {},and HAR_test_y length is {},\n'
                                         'in_target_HAR is {}and regressor_HAR is {},'
                                         'train_X is {} and train_y is {}'.format(len(in_target_HAR),len(regressor_HAR),\
                                                                                             len(HAR_test_X),len(HAR_test_y),\
                                                                                             in_target_HAR,regressor_HAR,\
                                                                                             train_X,train_y))

            else:
                if self.model_type == 1 or self.model_type == 3:
                    in_target_HAR = self.HAR_model().in_target[self.HAR_model().in_target.index.isin(train_y.index)]
                    regressor_HAR = self.HAR_model().regressor[self.HAR_model().regressor.index.isin(train_X.index)]
                self.train_X, self.train_y = train_X,train_y

        else:
            self.train_X, self.train_y = train_X, train_y

        def on_generation(ga_instance):
            # best_solution[0]是存着所有的solution
            # best_solution[1]是存着所有的fitness
            # best_solution[2]是存着最佳的solution的index
            fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            if self.if_print == True:

                print("")
                print("Generation = {generation}".format(generation=ga_instance.generations_completed))
                print("Fitness    = {fitness}".format(
                    fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
                print("Change     = {change}".format(
                    change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[
                               1] - self.last_fitness))
                self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

            if abs(fitness) <= 0.8 and self.Methodkeys == 'TSCV':

                # 跳出的条件，当fitness值小于0.8时，停止迭代
                # 仅当调用Time series cross-validator时才会使用

                return "stop"

        def fitness_function(solution,solution_idx):
            '''
            使用验证集数据来进行GA-SVR的参数优化

            '''
            if self.Methodkeys == 'No split':
                if self.useData == 'test':
                    if self.model_type == 1 or self.model_type == 3:
                        svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])
                        svr_prediction = svr_optimization.fit(X=self.train_X, y=self.train_y).predict(self.test_X)+ \
                                         self.HAR_model_fit(x=HAR_train_X, y=HAR_train_y).predict(HAR_test_X)

                        i = int(len(self.test_X)//testDatalengthRate)
                        fitness_functions = {
                            'MAE': mean_absolute_error,
                            'r2': r2_score,
                            'MAPE': mean_absolute_percentage_error,
                            'MSE': mean_squared_error
                        }

                        fitness_fn = fitness_functions[self.fitnessFunction]
                        fitness = fitness_fn(HAR_test_y, svr_prediction)
                        return -fitness if self.fitnessFunction in ['MAE', 'MAPE', 'MSE'] else fitness
                    #TODO:使用验证集即useData = test的时候的处理方法，运行ok，但是不知道有没有问题
                    if self.model_type == 2:
                        if len(self.train_X) < max(self.lags):
                            raise ValueError('length of training data is less than length of max lags of HAR model')
                        else:
                            svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])
                            svr_prediction = (1-solution[3])*svr_optimization.fit(X=self.train_X, y=self.train_y).predict(self.test_X)\
                                            +solution[3]*self.HAR_model_fit(x=self.train_X, y=self.train_y).predict(self.test_X)

                        fitness_functions = {
                            'MAE': mean_absolute_error,
                            'r2': r2_score,
                            'MAPE': mean_absolute_percentage_error,
                            'MSE': mean_squared_error
                        }

                        fitness_fn = fitness_functions[self.fitnessFunction]
                        if len(self.test_y)!=len(svr_prediction):
                            raise ValueError('length of test_y is not equal to length of svr_prediction,test_y is {},'
                                             'svr_prediction is {}'.format(self.test_y,svr_prediction))
                        #TODO:self.test_y 里面有一个NaN值，不知道怎么出现的。暂时先在get_SVR_train_data里面去掉了
                        #if self.test_y or svr_prediction contains nan value raise ValueError
                        # print('self.test_y is {}\nsvr_prediction is {}\n'
                        #       'train_X is \n{} train_y is {}\n'.format(self.test_y, svr_prediction,\
                        #                                          train_X,train_y))
                        # if np.isnan(self.test_y).any() or np.isnan(svr_prediction).any():
                        #     raise ValueError('test_y or svr_prediction contains nan value'
                        #                      'test_y is {},svr_prediction is {}'.format(self.test_y, svr_prediction))
                        fitness = fitness_fn(self.test_y, svr_prediction)
                        return -fitness if self.fitnessFunction in ['MAE', 'MAPE', 'MSE'] else fitness

                elif self.useData == 'train':
                    if self.model_type == 1 or self.model_type == 3:
                        svr_optimization = SVR(kernel= 'rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])

                        svr_prediction = svr_optimization.fit(X=self.train_X, y=self.train_y).predict(self.train_X)
                        # print('svr_prediction is {}'.format(svr_prediction))
                        i = int(len(self.test_X) * 2 / 3)
                        fitness_functions = {
                            'MAE': mean_absolute_error,
                            'r2': r2_score,
                            'MAPE': mean_absolute_percentage_error,
                            'MSE': mean_squared_error
                        }

                        fitness_fn = fitness_functions[self.fitnessFunction]
                        fitness = fitness_fn(self.train_y, svr_prediction)
                        return -fitness if self.fitnessFunction in ['MAE', 'MAPE', 'MSE'] else fitness

                    elif self.model_type == 2 :

                        svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])
                        svr_prediction = (1-solution[3])*svr_optimization.fit(X=self.train_X,\
                                                                              y=self.train_y).predict(self.train_X)\
                                        +solution[3]*self.HAR_model_fit().predict(self.train_X)

                        fitness_functions = {
                            'MAE': mean_absolute_error,
                            'r2': r2_score,
                            'MAPE': mean_absolute_percentage_error,
                            'MSE': mean_squared_error
                        }

                        fitness_fn = fitness_functions[self.fitnessFunction]
                        fitness = fitness_fn(self.train_y, svr_prediction)
                        return -fitness if self.fitnessFunction in ['MAE', 'MAPE', 'MSE'] else fitness


            elif self.Methodkeys == 'TSCV':

                svr_optimization = SVR(kernel='rbf', gamma=solution[0], C=solution[1], epsilon=solution[2])
                #为了实现one-day ahead的预测，所以分成了这么多份，因为预定传入的size是1000，所以分9份
                tscv = TimeSeriesSplit(test_size=1, gap=0, n_splits=3)

                ScoreOfmodel = 0
                count = 0
                for train_index, test_index in tscv.split(self.train_X):

                    #print("TRAIN:", train_index, "TEST:", test_index)

                    index_of_train = self.train_X.index[train_index]
                    index_of_test = self.train_X.index[test_index]

                    X_train, X_test = self.train_X.loc[index_of_train], self.train_X.loc[index_of_test]

                    y_train, y_test = self.train_y.loc[index_of_train], self.train_y.loc[index_of_test]
                    svr_prediction = svr_optimization.fit(X=X_train, y=y_train).predict(X_test)
                    #TODO: 2/15 21:27   完善了model_type 2的时间序列交叉验证
                    if self.model_type == 2:
                        svr_prediction = (1-solution[3])*svr_prediction+\
                            solution[3]*self.HAR_model_fit(x=X_train, y=y_train).predict(X_test)

                    #TODO: 2/15 21:25 增加了时间序列交叉验证的HAR模型
                    if self.model_type == 1 or self.model_type == 3:
                        train_har_x, train_har_y = regressor_HAR.loc[index_of_train], in_target_HAR.loc[index_of_train]
                        test_har_x, test_har_y = regressor_HAR.loc[index_of_test], in_target_HAR.loc[index_of_test]
                        har_pred = self.HAR_model_fit(x=train_har_x, y=train_har_y).predict(test_har_x)
                        svr_prediction = har_pred + svr_prediction
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

                fitness_dict = {
                    'MAE': -fitness,
                    'r2': fitness,
                    'MSE': -fitness,
                    'MAPE': -fitness
                }

                return fitness_dict[self.fitnessFunction]


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


        # best_solution[0]是存着所有的solution，best_solution[1]是存着所有的fitness，best_solution[2]是存着最佳的solution的index
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

        gasvr = SVR(kernel='rbf',gamma=solution[0],C=solution[1],epsilon=solution[2])

        return gasvr

    def get_alpha(self):
        '''
        only for model-type 2.
        result = alpha*HAR + (a-alpha)*SVR
        :return: alpha
        '''

        return self.solution[3]

    def predict(self, step=1):
        '''

        :param predictdata: 用来作预测的data
        :return: 预测出来的结果
        '''
        X, Y = self.get_SVR_data_train()
        gasvr = self.ga_svr()
        gasvr_fit = gasvr.fit(X=X,y=Y)

        har_model_fit = self.HAR_model_fit()
        # 下面的模型如果用observation,后面的时候会出现NaN，但是用observation-1的话会导致预测的日期不对。可能需要用observation+1
        har_model_1 = self.HAR_model(observation=self.observation)
        x = self.get_predict_data(har_model_1.predictor,days=step)

        if self.model_type == 1 or self.model_type == 3:
            if step != 1:
                raise ValueError('step must be 1 when model_type is 1')

            x_svr = self.get_SVR_data_test()
            x_svr.columns = x_svr.columns.astype(str)
            # if x_svr contains Nan value raise error.
            x_svr = x_svr.dropna(axis=0)
            if x_svr.isnull().values.any():
                # 训练集的X到5/29 （使用的data是0910 然后observation是100），所以预测出来的是6/1的，所以预测的时候第一天的是要用6/1去预测6/2，
                # 所以这个时候的NaN是在6/2里的，（因为6/2不在样本内，所以缺少residual）所以这里直接去掉Nan即可
                # TODO：确认HARmodel没有出现in-Sample的数据去预测的问题。
                raise ValueError('x_svr contains Nan value,x_svr is {},and x_svr.columns is {},'
                                 'X_train_svr is {},Y_train_svr is {}'.format(x_svr,x_svr.columns,\
                                                                              X,Y))

            #print('x_svr.columns is {},X.columns is {}'.format(x_svr.columns,X.columns))
            result = gasvr_fit.predict(x_svr)+har_model_1.fit().predict(x_svr.iloc[:,:3])

        elif self.model_type == 2:
            alpha = self.get_alpha()

            result = (1-alpha)*gasvr_fit.predict(x) + alpha*har_model_fit.predict(x)

        return result
