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
        result.append([os.path.abspath(os.path.join("Data/Lager5/complete_set/", entry)), 1 if entry in gt_list else 0])
    return result

roi_list = load_txt_list("roi/Lager5.txt")
gt_list = load_txt_list("gt/Lager5.txt")

test_list = roi_list[1::2]
test_data = expand_and_annotate_list(test_list, gt_list)

for i, (data_file, category) in enumerate(test_data):
	if "2019_03_16__14_52_12.mat" in data_file or "2019_03_16__14_54_11.mat" in data_file:
		gt_begin_index = i

def std_all_values(a):
    result = []
    for i in range(1, a.shape[0]):
        result.append(np.std(a[:i]))
    return np.asarray(result)


plot_data = np.load("classify_train_on_both_plot_data_on_bearing_5.npy")
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 9), dpi=200)

stds = std_all_values(plot_data[:,1].astype(np.float))

viridis = plt.cm.get_cmap('viridis')
colors = ["g" if x == 0 else "r" for x in plot_data[:,0].astype(np.float)]
# plt.scatter(x=np.arange(len(plot_data)), y=plot_data[:,1], c=viridis(plot_data[:,1]))
plt.ylim(0, 1)
plt.scatter(x=np.arange(len(plot_data)), y=plot_data[:,1].astype(np.float), c=colors)
plt.plot(np.arange(len(plot_data) - 1), stds, c="black")
ticks = plot_data[:,2][::12]
# ticks = [" ".join(x.split("__"))for x in ticks]
result_ticks = []
for days, times in [x.split("__") for x in ticks]:
	result_ticks.append(f"{days.replace('_', '.')} {times.replace('_', ':')}")
plt.xticks(np.arange(len(plot_data))[::12], result_ticks, rotation=25, ha="right")
plt.vlines(gt_begin_index, 0, 1, colors="black")
plt.hlines(y=0.5, linestyles="dashed", xmin=0, xmax=len(plot_data))
plt.title("Classification Bearing 5 (Trained on Bearing 4 and 5)")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("classify_plot_std_all_values.png")