import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def load_txt_list(path):
    result = []
    with open(path) as f:
        for line in f:
            result.append(line.lstrip().rstrip())

    return result

def expand_and_annotate_list(txt_list, gt_list):
    result = []
    for entry in txt_list:
        result.append([os.path.abspath(os.path.join("Data/Lager4/complete_set/", entry)), 1 if entry in gt_list else 0])
    return result

roi_list = load_txt_list("roi/Lager4.txt")
gt_list = load_txt_list("gt/Lager4.txt")

test_list = roi_list[1::2]
test_data = expand_and_annotate_list(test_list, gt_list)

for i, (data_file, category) in enumerate(test_data):
	if "2019_02_28__23_25_17.mat" in data_file:
		gt_begin_index = i

def windowed_data(a, stepsize, width):
	return np.vstack([a[i - width:i] for i in range(width, a.shape[0] + 1, stepsize)])

plot_data = np.load("classify_train_on_both_plot_data_on_bearing_4.npy")
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 9), dpi=200)

window_size = 20
w = windowed_data(plot_data[:,1].astype(np.float), 1, window_size)
stds = np.std(w, 1)

viridis = plt.cm.get_cmap('viridis')
colors = ["g" if x == 0 else "r" for x in plot_data[:,0].astype(np.float)]
# plt.scatter(x=np.arange(len(plot_data)), y=plot_data[:,1], c=viridis(plot_data[:,1]))
plt.ylim(0, 1)
plt.scatter(x=np.arange(len(plot_data)), y=plot_data[:,1].astype(np.float), c=colors)
plt.plot((window_size // 2) + np.arange(len(plot_data) - window_size + 1), stds, c="black")
ticks = plot_data[:,2][::6]
# ticks = [" ".join(x.split("__"))for x in ticks]
result_ticks = []
for days, times in [x.split("__") for x in ticks]:
	result_ticks.append(f"{days.replace('_', '.')} {times.replace('_', ':')}")
plt.xticks(np.arange(len(plot_data))[::6], result_ticks, rotation=25, ha="right")
plt.vlines(gt_begin_index, 0, 1, colors="black")
plt.hlines(y=0.5, linestyles="dashed", xmin=0, xmax=len(plot_data))
plt.title("Classification Bearing 4 (Trained on Bearing 4 and 5)")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("classify_plot_std.png")