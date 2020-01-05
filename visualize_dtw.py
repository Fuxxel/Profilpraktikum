import numpy as np
import matplotlib.pyplot as plt
import os 

filter_list = []
with open("Lager5_filter.txt") as ff:
	for line in ff:
		filter_list.append(line.lstrip().rstrip())

def is_allowed(name):
	return not name in filter_list


def trailing_moving_average(data, window_size):
	cumsum = np.cumsum(np.insert(data, 0, 0)) 
	return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


input_path = "dtw/Lager4/skip_29/normalized"
save_base = "dtw_lager4_skip_29_overview_normalized"
log = False
apply_filter = False
# cutoff_start = 2167 - (29*12)
# cutoff_end = 2167 + (29*12)

# cutoff_index = 323
downsample = 1

files = os.listdir(input_path)
files = filter(lambda x: x.endswith(".dtw"), files)
files = list(sorted(files))

dates = [".".join(x.split("_")[1:3][::-1]) for x in files]
ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(files, dates)][::downsample]

files = [os.path.join(input_path, x) for x in files]

dtw_data = []
for file in files[::downsample]:
	data = np.loadtxt(file)
	dtw_data.append(data)

dtw_data = np.array(dtw_data)

if apply_filter:
	filter_indices = []
	for i, label in enumerate(ticks):
		if not is_allowed(label):
			filter_indices.append(i)

	for label in filter_list:
		ticks.remove(label)

	dtw_data = np.delete(dtw_data, filter_indices, 0)

plt.figure(figsize=(11.27*12, 7.04), dpi=227)
# plt.title("DTW (2019_03_13__14_03_16.mat)") # Lager 5
plt.title("DTW (2019_02_28__12_04_16.mat)") # Lager 4
if log:
	plt.semilogy(dtw_data, label="Global error")
else:
	plt.plot(dtw_data, label="Global error")
	
plt.xticks(np.arange(len(ticks)), ticks)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")

plt.tight_layout() 
plt.savefig(save_base + ".png")
plt.clf()

dtw_data = trailing_moving_average(dtw_data, 10)

plt.figure(figsize=(11.27*12, 7.04), dpi=227)
# plt.title("DTW (2019_03_13__14_03_16.mat)") # Lager 5
plt.title("DTW (2019_02_28__12_04_16.mat)") # Lager 4
if log:
	plt.semilogy(dtw_data, label="Global error")
else:
	plt.plot(dtw_data, label="Global error")
	
plt.xticks(np.arange(len(ticks[10:])), ticks[10:])
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")

plt.tight_layout() 
plt.savefig(save_base + "_moving_average.png")
plt.clf()