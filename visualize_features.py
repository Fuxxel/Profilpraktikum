import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas

input_path = "features/Lager5"
save = "feature_plots/Lager5"
cutoff_index = 323

os.makedirs(save, exist_ok=True)

files = os.listdir(input_path)
files = filter(lambda x: x.endswith(".csv"), files)
files = list(sorted(files))

dates = [".".join(x.split("_")[1:3][::-1]) for x in files]
ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(files, dates)][:cutoff_index]

files = [os.path.join(input_path, x) for x in files]

num_cols = len(pandas.read_csv(files[0]).columns) - 1

metrics = [[] for _ in range(num_cols)]
col_names = None

for file in files[:cutoff_index]:
	data = pandas.read_csv(file)
	col_names = data.columns[1:]
	for i, col in enumerate(col_names):
		metrics[i].append(data[col].values[0])

for i in range(num_cols):
	plt.figure(figsize=(11.27*7, 7.04), dpi=227)
	plt.title(col_names[i])
	plt.plot(np.array(metrics[i]))
		
	plt.xticks(np.arange(len(ticks)), ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")

	plt.tight_layout() 
	plt.savefig(os.path.join(save, col_names[i] + ".png"))
	plt.clf()
	plt.close()
