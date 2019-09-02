from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

series = Series.from_csv("/Users/richardcollins/Desktop/Time_Series/daily-min-temperatures.csv", header=0)

# split dataset
X = series.values
days_to_predict = 7
train, test = X[1:len(X)-days_to_predict], X[len(X)-days_to_predict:]

# train autoregression
model = AR(train)
model_fit = model.fit()
print("Lag: {0}".format(model_fit.k_ar))
for i in model_fit.params:
    print("Coefficients: {:9.3f}".format(i))

# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print("predicted={0:.3g}, expected={1:.3g}".format(predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print("Test MSE: {0:.6g}".format(error))

# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
