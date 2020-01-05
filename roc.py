import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

filter_list = []
with open("Lager5_filter.txt") as ff:
	for line in ff:
		filter_list.append(line.lstrip().rstrip())

def is_allowed(name):
	return not name in filter_list

def trailing_moving_average(data, window_size):
	cumsum = np.cumsum(np.insert(data, 0, 0)) 
	return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

switch = "8"
apply_filter = False

apply_moving_average = True
moving_average_window = 10

# Lager 5
# data_file = "global_plots/Lager5/predicted_with_correct_net/skip_29/lager5_global_error_cnn_whole_dataset_skip_29_filt_{}.npy".format(switch)
# labels_file = "global_plots/Lager5/predicted_with_correct_net/skip_29/lager5_global_error_cnn_whole_dataset_skip_29_filt_{}.ticks".format(switch)

# Lager 4
data_file = "global_plots/Lager4/skip_0/lager4_global_error_cnn_whole_dataset_filt_{}.npy".format(switch)
labels_file = "global_plots/Lager4/skip_0/lager4_global_error_cnn_whole_dataset_filt_{}.ticks".format(switch)

# 2167
# cutoff_start = 2167 - (29*12)
# cutoff_end = 2167 + (29*12)
downsample = 1

metrics = np.load(data_file)[::downsample][:,2]
labels = pickle.load(open(labels_file, "rb"))[::downsample]

if apply_filter:
	filter_indices = []
	for i, label in enumerate(labels):
		if not is_allowed(label):
			filter_indices.append(i)

	for label in filter_list:
		if label in labels:
			labels.remove(label)

	metrics = np.delete(metrics, filter_indices, 0)

y = np.zeros_like(metrics)

# Lager 5
# y[int(metrics.shape[0] / 2):] = 1

# Lager 4
y[int(metrics.shape[0] / 2):] = 1

y = y[moving_average_window - 1:]

if apply_moving_average:
	metrics = trailing_moving_average(metrics, moving_average_window)
	labels = labels[moving_average_window - 1:]

fpr, tpr, thresholds = roc_curve(y, metrics)
roc_auc = roc_auc_score(y, metrics)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label="ROC curve (area = {:0.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()