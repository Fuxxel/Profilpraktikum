import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import linregress
from scipy.io import loadmat

lager = "Lager4"

input_folder = "Data/{}/complete_set".format(lager)
output_folder = "autocorrelation/{}".format(lager)

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(input_folder)
files = filter(lambda x: x.endswith(".mat"), files)
files = sorted(files)

for i, file in enumerate(files):
	print("{}/{}: {}".format(i + 1, len(files), file))
	filename = file.split(".")[0]
	data = loadmat(os.path.join(input_folder, file))["Data"][:, 0][:6144*50]
	
	plot_acf(data, lags=[(6144//4) * (x + 1) for x in range(4 * 4)])
	plt.savefig(os.path.join(output_folder, filename + ".png"))
	plt.clf()