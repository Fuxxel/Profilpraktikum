import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

base_path = "classify_shift_results_bearing_5"

runs = range(0, 20)
sensor = 0
shifts = range(-15, 16)

results = []
for shift in shifts:	
	shift_result = []
	for run in runs:
		path_to_file = os.path.join(base_path, str(run), str(sensor), str(shift), "result.txt")
		with open(path_to_file, "r") as result_file:
			line = result_file.readline()
			precentage = float(line.split(":")[-1].lstrip().rstrip()[:-1])
			shift_result.append(precentage)
	results.append(shift_result)

medians = np.median(np.asarray(results), 1)

matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(16, 9), dpi=200)
plt.boxplot(results, showfliers=False)
plt.plot(np.arange(1, 1 + len(results)), medians, color="red", alpha=0.25, marker="x")
plt.xticks(np.arange(1, 1 + len(results)), shifts)
ax = plt.gca()
plt.vlines(16, *ax.get_ylim(), alpha=0.2, linestyles="dashed")
plt.title("Shift results (Bearing 5)")
plt.xlabel("Shift")
plt.ylabel("Classification Accuracy")
plt.savefig("classify_shift_plot_bearing_5.png")