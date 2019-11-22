import os
import pandas
import matplotlib.pyplot as plt
import numpy as np

source = "model_results/Lager4/enc_dec/lat_100/"

files = os.listdir(source)
files = sorted(filter(lambda x: x.endswith(".csv"), files))

metrics = []

for file in files:
	d = pandas.read_csv(os.path.join(source, file))[["Abs. Error","Sq. Error","Global"]]
	last_row = d.iloc[-1]
	metrics.append(last_row.values)

metrics = np.array(metrics)
x = np.arange(metrics.shape[0])

def norm(series):
	max = np.max(series)
	min = np.min(series)
	return (series - min) / (max - min)

metrics[:, 0] = norm(metrics[:, 0])
metrics[:, 1] = norm(metrics[:, 1])
metrics[:, 2] = norm(metrics[:, 2])

plt.plot(x, metrics[:, 0])
plt.plot(x, metrics[:, 1])
plt.plot(x, metrics[:, 2])

plt.show()
