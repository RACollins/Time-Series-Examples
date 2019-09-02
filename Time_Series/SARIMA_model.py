import ast
from math import sqrt
from pandas import Series
from matplotlib import pyplot
from itertools import product
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    model = SARIMAX(history,
                    order = order,
                    seasonal_order = sorder,
                    trend = trend,
                    enforce_stationarity = False,
                    enforce_invertibility = False)
    model_fit = model.fit(disp = False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return(yhat[0])

# root mean squared error
def measure_rmse(actual, predicted):
    return(sqrt(mean_squared_error(actual, predicted)))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return(data[:-n_test], data[-n_test:])

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    error = measure_rmse(test, predictions)
    return(error, test, predictions)

# score a model, return None on failure
def score_model(data, n_test, cfg, debug = False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result, _, __ = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result, _, __ = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(" > SARIMA{0} MSE = {1:.3g}".format(key, result))
    return((key, result))

# grid search configs
def grid_search(data, cfg_list, n_test, parallel = True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs = cpu_count(), backend = "multiprocessing")
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key = lambda tup: tup[1])
    return(scores)

# create a set of sarima configs to try
def sarima_configs(seasonal = [0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    m_params = seasonal
    # create config instances
    for trend_elements in product(p_params, d_params, q_params):
        for t in t_params:
            for seasonal_elements in product(P_params, D_params, Q_params, m_params):
                cfg2 = [trend_elements, seasonal_elements, t]
                models.append(cfg2)
    return(models)

if __name__ == '__main__':
    # define dataset
    series = Series.from_csv("/Users/richardcollins/Desktop/Time_Series/monthly-car-sales.csv", header = 0, index_col = 0)
    data = series.values
    print(data.shape)
    # data split
    n_test = 12
    # model configs
    cfg_list = sarima_configs(seasonal = [12])
    # grid search
    scores = grid_search(data, cfg_list, n_test, parallel = False)
    print("done")
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)

    train, test = train_test_split(data, n_test)
    best_params = ast.literal_eval(scores[0][0])
    best_error, best_test, best_predictions = walk_forward_validation(data, n_test, best_params)

    # plot
    pyplot.plot(best_test)
    pyplot.plot(best_predictions, color = "red")
    pyplot.show()
