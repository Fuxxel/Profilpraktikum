import pandas
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

lager = "Lager5"

def load_txt_list(path):
	result = []
	with open(path, "r") as txt_file:
		for line in txt_file:
			result.append(line.rstrip().lstrip())
	return result

source_path = "useful_features/{}".format(lager)

files = os.listdir(source_path)
files = filter(lambda x: x.endswith(".csv"), files)
files = sorted(files)

metrics = []
for file in files:
	full_file_path = os.path.join(source_path, file)

	data = pandas.read_csv(full_file_path)
	first_row = data.iloc[0]
	metrics.append([first_row["x__autocorrelation__lag_6144"], first_row["x__mean_abs_change"]])

metrics = np.array(metrics)

files = list(map(lambda x: x.replace("csv", "mat"), files))

dates = [".".join(x.split("_")[1:3][::-1]) for x in files]
ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(files, dates)]

roi_list = load_txt_list(os.path.join("roi", lager + ".txt"))
joined = zip(metrics, ticks, files)
filtered_joined = filter(lambda x: x[2] in roi_list, joined)

roi_metrics, roi_ticks, roi_folders = zip(*filtered_joined) # Unzip
roi_metrics = np.array(roi_metrics)

plt.figure(figsize=(11.27*12, 7.04), dpi=227)
plt.title("Autocorrelation lag 6144")
plt.plot(roi_metrics[:, 0])
if len(roi_ticks) > 500:
	plt.xticks(np.arange(len(roi_ticks))[::len(roi_ticks)//500], roi_ticks[::len(roi_ticks)//500])
else:
	plt.xticks(np.arange(len(roi_ticks)), roi_ticks)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")
plt.xlabel("Date")
plt.ylabel("Value")
plt.tight_layout() 
plt.savefig("useful_features/{}/autocorrelation.png".format(lager))
plt.clf()

plt.figure(figsize=(11.27*12, 7.04), dpi=227)
plt.title("Mean abs change")
plt.plot(roi_metrics[:, 1])
if len(roi_ticks) > 500:
	plt.xticks(np.arange(len(roi_ticks))[::len(roi_ticks)//500], roi_ticks[::len(roi_ticks)//500])
else:
	plt.xticks(np.arange(len(roi_ticks)), roi_ticks)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")
plt.xlabel("Date")
plt.ylabel("Value")
plt.tight_layout() 
plt.savefig("useful_features/{}/mean_abs_change.png".format(lager))
plt.clf()

names = ["autocorrelation", "mean_abs_change"]
for i, name in enumerate(names):
	gt_list = load_txt_list(os.path.join("gt", lager + ".txt"))
	last_sample_in_gt = gt_list[-1]
	first_sample_in_gt = gt_list[0]

	joined = zip(metrics[:, i], ticks, files)
	filtered_joined = filter(lambda x: x[2] <= last_sample_in_gt, joined)

	roc_metrics, roc_ticks, roc_folders = zip(*filtered_joined) # Unzip
	roc_metrics = np.array(roc_metrics)

	gt_begin_index = roc_folders.index(first_sample_in_gt)
	gt_horizontal_line_y_value = roc_metrics[gt_begin_index]

	gt_labels = []
	for folder in roc_folders:
		if name == "autocorrelation": # Reverse classifier for autocorrelation because autocorrelation goes down if error goes up
			gt_labels.append(0 if folder in gt_list else 1)
		else:
			gt_labels.append(1 if folder in gt_list else 0)

	assert(len(gt_labels) == len(roc_ticks))
	if name == "autocorrelation":
		assert(len(gt_labels) - sum(gt_labels) == len(gt_list))
	else:
		assert(sum(gt_labels) == len(gt_list))

	fpr, tpr, thresholds = roc_curve(gt_labels, roc_metrics)
	roc_auc = roc_auc_score(gt_labels, roc_metrics)
	if roc_auc < 0.5:
		# Reverse the classifier
		gt_labels = list(map(lambda x: 1 - x, gt_labels))
		fpr, tpr, thresholds = roc_curve(gt_labels, roc_metrics)
		roc_auc = roc_auc_score(gt_labels, roc_metrics)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			lw=lw, label="ROC curve (area = {:0.2f})".format(roc_auc))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	num = 0
	colors = iter(["r*", "g*", "y*"])
	threshold_levels = []
	for fpos, tpos, thresh in zip(fpr, tpr, thresholds):
		dist = np.sum((np.array([0, 1]) - np.array([fpos, tpos]))**2)
		threshold_levels.append((dist, fpos, tpos, thresh))

	threshold_levels = sorted(threshold_levels, key=lambda x: x[0])

	hline_thresholds = []
	for i in range(3):
		_, fpos, tpos, tresh = threshold_levels[i]
		hline_thresholds.append(tresh)
		plt.plot(fpos, tpos, next(colors))

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.tight_layout() 
	plt.savefig("useful_features/{}/roc_{}_annotated.png".format(lager, name))
	plt.clf()

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(lager)
	plt.plot(roc_metrics)
	plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	plt.hlines(hline_thresholds[0], 0, roc_metrics.shape[0], colors=["red"], linestyles="dashdot")
	plt.hlines(hline_thresholds[1], 0, roc_metrics.shape[0], colors=["green"], linestyles="dashdot")
	plt.hlines(hline_thresholds[2], 0, roc_metrics.shape[0], colors=["yellow"], linestyles="dashdot")
	plt.xlabel("Date")
	plt.ylabel("Value")
	if len(roc_ticks) > 500:
		plt.xticks(np.arange(len(roc_ticks))[::len(roc_ticks)//500], roc_ticks[::len(roc_ticks)//500])
	else:
		plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig("useful_features/{}/roc_metric_{}_annotated.png".format(lager, name))
	plt.clf()

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
	plt.savefig("useful_features/{}/roc_{}.png".format(lager, name))
	plt.clf()

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(lager)
	plt.plot(roc_metrics)
	plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	plt.xlabel("Date")
	plt.ylabel("Value")
	if len(roc_ticks) > 500:
		plt.xticks(np.arange(len(roc_ticks))[::len(roc_ticks)//500], roc_ticks[::len(roc_ticks)//500])
	else:
		plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig("useful_features/{}/roc_{}_metric.png".format(lager, name))
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
	plt.savefig("useful_features/{}/pr_{}.png".format(lager, name))
	plt.clf()
