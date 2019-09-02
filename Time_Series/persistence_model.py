from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

series = read_csv("/Users/richardcollins/Desktop/Time_Series/daily-min-temperatures.csv", header=0)

# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values[1].shift(1), values[1]], axis=1)
dataframe.columns = ["t-1", "t"]

# Check correlation at lag = 1
corr, _ = pearsonr(dataframe["t-1"][1:], dataframe["t"][1:])
print("Lag-1 correlation: {0:.6g}".format(corr))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return(x)

# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print("Test MSE: {0:.6g}".format(test_score))

# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
