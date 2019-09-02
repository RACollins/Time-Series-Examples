from pandas import Series
from matplotlib import pyplot
from pandas import DataFrame
from pandas import TimeGrouper
from pandas import concat
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

series = Series.from_csv("/Users/richardcollins/Desktop/Time_Series/daily-min-temperatures.csv", header=0)
print(series.head())
print(len(series))

# Group data by years and by months (in 1990)
groups = series.groupby(TimeGrouper('A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
series_1990 = series['1990']
groups_1990 = series_1990.groupby(TimeGrouper('M'))
months = concat([DataFrame(x[1].values) for x in groups_1990], axis=1)
months = DataFrame(months)
months.columns = range(1, 13)


# Line plot
series.plot(linewidth = 0.2)
pyplot.show()

# Line plot per year
years.plot(subplots=True, legend=False)
pyplot.show()

# Histogram
series.hist()
pyplot.show()

# Density plot
series.plot(kind='kde')
pyplot.show()

# Box and whisker plot per year
years.boxplot()
pyplot.show()

# Box and whisker plot per month
months.boxplot()
pyplot.show()

# Heatmap plot per year
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()

# Heatmap plot per month
pyplot.matshow(months, interpolation=None, aspect='auto')
pyplot.show()

# Lag plot
lag_plot(series)
pyplot.show()

# Autocorrelation plot
autocorrelation_plot(series)
pyplot.show()
