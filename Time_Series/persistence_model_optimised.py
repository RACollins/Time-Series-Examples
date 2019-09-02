import numpy as np
from pandas import read_csv
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

series = Series.from_csv("/Users/richardcollins/Desktop/Time_Series/daily-min-temperatures.csv", header=0)

# prepare data
X = series.values
days_to_predict = 7
train, test = X[1:len(X)-days_to_predict], X[len(X)-days_to_predict:]

# find optimum persistence value
persistence_values = range(1, len(X)-days_to_predict)
scores = list()
for p in persistence_values:
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# make prediction
		yhat = history[-p]
		predictions.append(yhat)
		# observation
		history.append(test[i])
	# report performance
	rmse = sqrt(mean_squared_error(test, predictions))
	scores.append(rmse)
	#print("p={0} RMSE:{1:.6g}".format(p, rmse))

min_lag_index = np.argmin(scores)
min_lag = min_lag_index + 1

print("Best lag value: t-{0}".format(min_lag))
print("RMSE at best lag: {0:.6}".format(scores[min_lag_index]))

# plot scores over persistence values
#pyplot.plot(persistence_values, scores)
#pyplot.show()

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	yhat = history[-min_lag]
	predictions.append(yhat)
	# observation
	history.append(test[i])
# plot predictions vs observations
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
