import warnings
from pandas import Series
from pandas import DataFrame
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def evaluate_arima_model(X, arima_order, days_to_predict = 7):
    train, test = X[1:len(X)-days_to_predict], X[len(X)-days_to_predict:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp = 0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    return(error, test, predictions)

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype("float32")
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse, _, __ = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print("ARIMA{0}, MSE = {1:.3g}".format(order, mse))
                except:
                    continue
    print("Best ARIMA{0}, MSE = {1:.3g}".format(best_cfg, best_score))
    return(best_cfg)

if __name__ == '__main__':
    # load data
    series = Series.from_csv("/Users/richardcollins/Desktop/Time_Series/shampoo.csv", header = 0)
    X = series.values

    '''# check autocorrelation plot
    autocorrelation_plot(series)
    pyplot.xlim(right=30)
    pyplot.show()

    # fit model
    model = ARIMA(series, order=(10, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())'''

    # evaluate parameters
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")

    best_params = evaluate_models(X, p_values, d_values, q_values)
    best_error, best_test, best_predictions = evaluate_arima_model(X, best_params)

    # plot
    pyplot.plot(best_test)
    pyplot.plot(best_predictions, color="red")
    pyplot.show()
