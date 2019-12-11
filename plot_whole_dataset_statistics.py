import matplotlib.pyplot as plt
import pickle
import numpy as np

switch = "32"
log = False

data_file = "lager5_global_error_cnn_whole_dataset_skip_29_filt_{}.npy".format(switch)
labels_file = "lager5_global_error_cnn_whole_dataset_skip_29_filt_{}.ticks".format(switch)

save = "lager5_global_error_cnn_whole_dataset_filt_{}_around_error".format(switch)

if log:
	save += "_log.png"
else:
	save += ".png"

# 2167
cutoff_start = 2167 - (29*12)
cutoff_end = 2167 + (29*12)
downsample = 1

metrics = np.load(data_file)[cutoff_start:cutoff_end:downsample]
labels = pickle.load(open(labels_file, "rb"))[cutoff_start:cutoff_end:downsample]

plt.figure(figsize=(11.27*12, 7.04), dpi=227)
# plt.plot(metrics[:,0], label="Abs error")
# plt.plot(metrics[:,1], label="Sq error")
# plt.legend()
# plt.title("Abs. and sq. error")
plt.title("Global error")
# plt.locator_params(axis="x", nbins=len(folders))
if log:
	plt.semilogy(metrics[:,2], label="Global error")
else:
	plt.plot(metrics[:,2], label="Global error")
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
