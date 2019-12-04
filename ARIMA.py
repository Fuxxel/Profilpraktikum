from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat

window_size = 1024

path = "Sample Data/2019_02_28__10_37_15.mat"

data = loadmat(path)["Data"][:,0]

test = data[window_size:]

predictions = []

for t in range(len(test)):
	print("{}/{}".format(t + 1, len(test)), end="")
	model = ARIMA(data[t:t + window_size], order=(5,1,5))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()[0]
	predictions.append(output)


print("")
