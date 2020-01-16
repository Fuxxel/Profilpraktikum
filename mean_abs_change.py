import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def load_txt_list(path):
	result = []
	with open(path, "r") as txt_file:
		for line in txt_file:
			result.append(line.rstrip().lstrip())
	return result

def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))

def trailing_moving_average(data, window_size):
	cumsum = np.cumsum(np.insert(data, 0, 0)) 
	return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def create_metrics(data_folder, lager, title, save):
	files = os.listdir(data_folder)
	files = list(filter(lambda x: x.endswith(".mat"), files))
	assert(len(files) > 0), "No .mat files found in data folder: {}".format(data_folder)

	print("Found {} .mat files in {}".format(len(files), data_folder))

	files = list(sorted(files))

	folders = files

	dates = [".".join(x.split("_")[1:3][::-1]) for x in folders]
	ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(folders, dates)]

	# Expand all files to full path
	files = list(map(lambda x: os.path.join(data_folder, x), files))

	metrics = []
	for i, file in enumerate(files):
		print("Processing {}/{}: {}".format(i + 1, len(files), file))
		data = loadmat(file)["Data"][..., 0] # Take zero'th timeseries
		metrics.append(mean_abs_change(data))

	assert(len(ticks) == len(metrics))
	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.plot(metrics)
	plt.title(title)
	plt.xlabel("Date")
	plt.ylabel("Mean abs error")
	plt.xticks(np.arange(len(ticks)), ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(save, "error.png"))
	plt.clf()

	moving_average_window = 20

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.plot(ma_metrics)
	plt.title(title)
	plt.xlabel("Date")
	plt.ylabel("Mean abs error")
	plt.xticks(np.arange(len(ma_ticks)), ma_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(save, "error_ma_20.png"))
	plt.clf()

	moving_average_window = 10

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(title)
	plt.plot(ma_metrics)
	plt.xlabel("Date")
	plt.ylabel("Mean abs error")
	plt.xticks(np.arange(len(ma_ticks)), ma_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(save, "error_ma_10.png"))
	plt.clf()

	#########################################
	#### Create ROC and AUC inside of ROI
	#########################################
	gt_list = load_txt_list(os.path.join("gt", lager + ".txt"))
	last_sample_in_gt = gt_list[-1]
	first_sample_in_gt = gt_list[0]
	
	joined = zip(metrics, ticks, folders)
	filtered_joined = filter(lambda x: x[2] <= last_sample_in_gt, joined)

	roc_metrics, roc_ticks, roc_folders = zip(*filtered_joined) # Unzip
	roc_metrics = np.array(roc_metrics)

	gt_begin_index = roc_folders.index(first_sample_in_gt)
	gt_horizontal_line_y_value = roc_metrics[gt_begin_index]

	gt_labels = []
	for folder in roc_folders:
		gt_labels.append(1 if folder in gt_list else 0)

	assert(len(gt_labels) == len(roc_ticks))
	assert(sum(gt_labels) == len(gt_list))

	fpr, tpr, thresholds = roc_curve(gt_labels, roc_metrics)
	roc_auc = roc_auc_score(gt_labels, roc_metrics)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			lw=lw, label="ROC curve (area = {:0.2f})".format(roc_auc))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.tight_layout() 
	plt.savefig(os.path.join(save, "roc.png"))
	plt.clf()

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(title)
	plt.plot(roc_metrics)
	plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	plt.xlabel("Date")
	plt.ylabel("Global error")
	plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(save, "roc_metric.png"))
	plt.clf()

	precision, recall, thresholds = precision_recall_curve(gt_labels, roc_metrics)
	average_precision = average_precision_score(gt_labels, roc_metrics)
	lw = 2
	plt.figure()
	plt.plot(recall, precision,
			lw=lw, label="Average Precision = {:0.2f}".format(average_precision))
	plt.xlim([0.0, 1.0])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve')
	plt.legend(loc="lower left")
	plt.tight_layout() 
	plt.savefig(os.path.join(save, "pr.png"))
	plt.clf()

def main():
	create_metrics("Data/Lager4/complete_set", "Lager4", "Lager 4 Mean Abs Change Error", "mean_abs_change/Lager4")
	create_metrics("Data/Lager5/complete_set", "Lager5", "Lager 5 Mean Abs Change Error", "mean_abs_change/Lager5")

if __name__ == "__main__":
	main()

