import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

run = 3
lager = 4

file_with_plot_data = f"classify_ims_transfer_from_run_2_bearing_1_to_run_{run}_bearing_{lager}.npy"

def find_gt_begin_index(data):
	for i, row in enumerate(data):
		if int(row[2]) == 1:
			return i
	return None

def std_all_values(a):
    result = []
    for i in range(1, a.shape[0]):
        result.append(np.std(a[:i]))
    return np.asarray(result)


plot_data = np.load(file_with_plot_data)
preds = 1.0 - plot_data[:,1].astype(np.float)
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 9), dpi=200)

window_size = 20
stds = std_all_values(preds)

viridis = plt.cm.get_cmap('viridis')
colors = ["r" if x < 0.5 else "g" for x in preds]
plt.ylim(0, 1)
plt.scatter(x=np.arange(len(plot_data)), y=preds, c=colors)
plt.plot(np.arange(len(plot_data) - 1), stds, c="black")

if run == 1:
	# tick_skip = 33 # For non-transfer predictions
	tick_skip = 66 # For transfer predictions
elif run == 2:
	# tick_skip = 16 # For non-transfer predictions
	tick_skip = 32 # For transfer predictions
elif run == 3:
	# tick_skip = 33*3 # For non-transfer predictions
	tick_skip = 66*3 # For transfer predictions

plt.xticks(np.arange(len(plot_data))[::tick_skip], plot_data[:,0][::tick_skip], rotation=45, ha="right")

begin_index = find_gt_begin_index(plot_data)
if begin_index:
	plt.vlines(begin_index, 0, 1, colors="black")

plt.hlines(y=0.5, linestyles="dashed", xmin=0, xmax=len(plot_data))
plt.title(f"Classification Run {run} Bearing {lager} (Trained on Run 2 Bearing 1)")
plt.xlabel("Date")
plt.ylabel("Predicted Bearing Health")
plt.tight_layout()
plt.savefig(f"classify_plot_ims_transfer_div_total_run_2_bearing_1_to_run_{run}_bearing_{lager}.png")