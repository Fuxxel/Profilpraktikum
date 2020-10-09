import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

metrics = np.load("plot_data.npy")
metrics = metrics[:, 2]
ticks = pickle.load(open("plot_data.ticks", "rb"))

gt_line_time = "16.03. 14:52:12"
gt_line_index = ticks.index(gt_line_time)
new_end = "16.03. 22:30:13"
end_index = ticks.index(new_end)

matplotlib.rcParams.update({'font.size': 16})
plt.figure(figsize=(16,9), dpi=200)
plt.plot(metrics[:end_index])

skip = 36
plt.xticks(np.arange(len(ticks[:end_index]))[::skip], ticks[:end_index:skip])
plt.vlines(gt_line_index, metrics[:end_index].min(), metrics[:end_index].max(), color="r")

plt.ylabel("Error")
plt.xlabel("Date")
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")
plt.tight_layout() 
plt.savefig("metric.png")