import os
import numpy as np

path = "model_predictions/Lager5/ws128/nc1/fs3/skip_first_0/lat_2"

def load_txt_list(path):
	result = []
	with open(path, "r") as txt_file:
		for line in txt_file:
			result.append(line.rstrip().lstrip())
	return result

folders = os.listdir(path)
folders = filter(lambda x: x.endswith(".mat"), folders)
folders = sorted(folders)
print("Found {} .mat files to create plots.".format(len(folders)))

roi_list = load_txt_list(os.path.join("roi", "Lager5.txt"))
filtered_joined = filter(lambda x: x in roi_list, folders)

metrics = []

for folder in filtered_joined:
	folder_path = os.path.join(path, folder)
	gt_path = os.path.join(folder_path, "complete_timeseries_input.npy")
	pred_path = os.path.join(folder_path, "complete_timeseries_predicted.npy")

	gt = np.load(gt_path) 
	pred = np.load(pred_path)

	abs_error = np.quantile(np.abs(gt - pred), 0.995)
	sq_error = np.quantile((gt - pred)**2, 0.995)
	global_error = np.sum((gt - pred)**2)

	metrics.append([abs_error, sq_error, global_error])

	print("{}: abs:{:.4f}, sq:{:.4f}, global:{:.4f}".format(folder, abs_error, sq_error, global_error))

metrics = np.array(metrics)[:, 2]
np.save("array_anton_lager_5.npy", metrics)