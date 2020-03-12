import numpy as np
import matplotlib.pyplot as plt

def normalize(d):
	return (d - np.min(d)) / (np.max(d) - np.min(d))

def mean_abs_change(x):
	return np.mean(np.abs(np.diff(x)))


def sigmoid(x, x_0, k, m):
	return m / (1 + np.exp(-k*(x - x_0)))
x = np.linspace(0, np.pi * 3, 10000)

y = np.sin(x)
y += np.sin(2*x*np.pi)
y += np.sin(3*x*np.pi)
y += np.sin(4*x*np.pi)
y += np.sin(5*x*np.pi)

error = np.sin(12*x*np.pi)

# alpha = 1/5

# error_amp = normalize(1/(np.sqrt(np.pi)*alpha) * np.exp(-((x - (np.pi * 3/2))**2)/(alpha**2)))
# y += error * error_amp

y = np.concatenate([np.ones(5000), x[:5000] + 1])

# x_0 = x[5000]

# y += error * sigmoid(x, x_0=x_0, k=1, m=1)
# y += sigmoid(x, x_0=x_0, k=1/2, m=10)

# print(mean_abs_change(y))

plt.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(16, 9), dpi=80 )

plt.plot(y)
plt.xlabel("Time")
plt.ylabel("Metric")

frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
frame.axes.yaxis.set_ticklabels([])
plt.tight_layout()
plt.show()