import numpy as np
import matplotlib.pyplot as plt
import os 

input_path = "dtw/Lager5"
save = "dtw_lager5.png"
log = False
cutoff_index = 323

files = os.listdir(input_path)
files = filter(lambda x: x.endswith(".dtw"), files)
files = list(sorted(files))

dates = [".".join(x.split("_")[1:3][::-1]) for x in files]
ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(files, dates)][:cutoff_index]

files = [os.path.join(input_path, x) for x in files]

dtw_data = []
for file in files[:cutoff_index]:
	data = np.loadtxt(file)
	dtw_data.append(data)

dtw_data = np.array(dtw_data)

plt.figure(figsize=(11.27*7, 7.04), dpi=227)
plt.title("DTW")
if log:
	plt.semilogy(dtw_data, label="Global error")
else:
	plt.plot(dtw_data, label="Global error")
	
plt.xticks(np.arange(len(ticks)), ticks)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")

plt.tight_layout() 
plt.savefig(save)
plt.clf()