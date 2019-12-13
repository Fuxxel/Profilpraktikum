import matplotlib.pyplot as plt
import pickle
import numpy as np

filter_list = []
with open("Lager5_filter.txt") as ff:
	for line in ff:
		filter_list.append(line.lstrip().rstrip())

def is_allowed(name):
	return not name in filter_list

def trailing_moving_average(data, window_size):
	cumsum = np.cumsum(np.insert(data, 0, 0)) 
	return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

switch = "32"
log = False
apply_filter = False

apply_moving_average = True
moving_average_window = 20

data_file = "global_plots/Lager5/predicted_with_correct_net/skip_29/lager5_global_error_cnn_whole_dataset_skip_29_filt_{}.npy".format(switch)
labels_file = "global_plots/Lager5/predicted_with_correct_net/skip_29/lager5_global_error_cnn_whole_dataset_skip_29_filt_{}.ticks".format(switch)

# overview plots
# save = "lager5_global_error_cnn_whole_dataset_filt_{}_overview".format(switch)

# around error plot
save = "lager5_global_error_cnn_whole_dataset_filt_{}_around_error_window_20".format(switch)

if log:
	save += "_log.png"
else:
	save += ".png"

# 2167
cutoff_start = 2167 - (29*12)
cutoff_end = 2167 + (29*12)
downsample = 1

metrics = np.load(data_file)[cutoff_start:cutoff_end:downsample][:,2]
labels = pickle.load(open(labels_file, "rb"))[cutoff_start:cutoff_end:downsample]

if apply_filter:
	filter_indices = []
	for i, label in enumerate(labels):
		if not is_allowed(label):
			filter_indices.append(i)

	for label in filter_list:
		labels.remove(label)

	metrics = np.delete(metrics, filter_indices, 0)

if apply_moving_average:
	metrics = trailing_moving_average(metrics, moving_average_window)
	labels = labels[moving_average_window:]

plt.figure(figsize=(11.27*12, 7.04), dpi=227)
# plt.plot(metrics[:,0], label="Abs error")
# plt.plot(metrics[:,1], label="Sq error")
# plt.legend()
# plt.title("Abs. and sq. error")
plt.title("Global error")
# plt.locator_params(axis="x", nbins=len(folders))
if log:
	plt.semilogy(metrics, label="Global error")
else:
	plt.plot(metrics, label="Global error")
plt.xticks(np.arange(len(labels)), labels)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")
# axes = plt.gca()
# axes.set_xticklabels([":".join(x.split("_")[-3:])[:-4] for x in folders])
# plt.show()
plt.tight_layout() 
plt.savefig(save)
plt.clf()
