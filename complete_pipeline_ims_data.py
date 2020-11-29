from models.keras.cnn import build_cnn_model
from models.keras.enc_dec import build_enc_dec_model, build_enc_dec_model_test
from dataloader import CNN_DataLoader, CNN_Test_DataLoader, LSTM_DataLoader, LSTM_Test_DataLoader

import argparse
import os
import numpy as np
import pickle
import glob
import ntpath
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf

def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

def normalize_window(window):
	min = window.min()
	return (window - min) / (window.max() - min)

def load_ims_data_training(path, window_size):
	cutoff = 13 if "Run1" in path else 6 # First hour corresponds to the first 13 files in Run1 else 6 for Run2 and Run3
	files = sorted(glob.glob(os.path.join(path, "*.npy")))[:cutoff] 
	complete_data = []
	for file in files:
		file_data = np.load(file)
		complete_data.append(file_data)

	complete_data = np.hstack(complete_data).T
	print(complete_data.shape)
	if "Run1" in path:
		complete_data = complete_data[:,0]

	windowed_data = []
	current_window_index = 0
	increment = window_size // 2
	while current_window_index < complete_data.shape[0] - window_size:
		window = normalize_window(complete_data[current_window_index:current_window_index + window_size])
		windowed_data.append(window[..., np.newaxis])
		current_window_index += increment

	return np.array(windowed_data)

def load_ims_data_test(path):
	begin = 13 if "Run1" in path else 6 # First hour corresponds to the first 13 files in Run1 else 6 for Run2 and Run3
	files = sorted(glob.glob(os.path.join(path, "*.npy")))[begin:] 
	complete_data = []
	for file in files:
		file_data = np.load(file)
		if "Run1" in path:
			file_data = file_data[0,:]
		complete_data.append(file_data)

	return zip(complete_data, files)

def load_txt_list(path):
	result = []
	with open(path, "r") as txt_file:
		for line in txt_file:
			result.append(line.rstrip().lstrip())
	return result

def trailing_moving_average(data, window_size):
	cumsum = np.cumsum(np.insert(data, 0, 0)) 
	return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def build_subpath_from_model_params(args):
	return os.path.join(args.name, "reduced_model" if args.use_reduced_model else "full_model", f"run_{args.run}", f"lager_{args.lager}", "ws{}".format(args.window_size), "nc{}".format(args.num_conv), "fs{}".format(args.filter_size), "skip_first_{}".format(args.skip_train_files), "lat_{}".format(args.filter_ratio))
	
def write_train_status_file(status, save_path):
	with open(save_path, "w") as out:
		for k, v in status.items():
			line = f"{k}:{v}\n"
			out.write(line)

def read_train_status_file(read_path):
	status = dict()
	with open(read_path, "r") as inp:
		for line in inp:
			split = line.split(":")
			k = split[0].lstrip().rstrip()
			v = split[1].lstrip().rstrip()
			status[k] = v
	
	return status

def train(model, args):
	print("##########################")
	print("Starting training")
	print("##########################")
	save_file_name = "cnn_model.h5"

	os.makedirs(os.path.join(args.save_model, build_subpath_from_model_params(args)), exist_ok=True)
	model_save_path = os.path.join(args.save_model, build_subpath_from_model_params(args), save_file_name)

	training_data = load_ims_data_training(args.training_input_path, args.window_size)
	x_train, x_val = train_test_split(training_data, test_size=0.1)

	print(training_data.shape)
	
	t_x = tf.data.Dataset.from_tensor_slices(x_train)
	t_y = tf.data.Dataset.from_tensor_slices(x_train)
	d_t = tf.data.Dataset.zip((t_x, t_y))
	b_t = d_t.shuffle(args.batch_size * 2).batch(args.batch_size)

	v_x = tf.data.Dataset.from_tensor_slices(x_val)
	v_y = tf.data.Dataset.from_tensor_slices(x_val)
	d_v = tf.data.Dataset.zip((v_x, v_y))
	b_v = d_v.batch(args.batch_size)

	model.fit(b_t, use_multiprocessing=(args.workers != 0), workers=args.workers, epochs=args.epochs, validation_data=b_v)

	print("Saving model to: {}".format(model_save_path))
	model.save(model_save_path)
	
	return model

def predict(model, args):
	print("##########################")
	print("Starting prediction")
	print("##########################")

	test_data = load_ims_data_test(args.training_input_path)

	for data, file_path in test_data:
		filename = path_leaf(file_path)

		save_path = os.path.join(args.save_predict, build_subpath_from_model_params(args), filename)
		os.makedirs(save_path, exist_ok=True)

		windowed_data = []
		current_window_index = 0
		increment = args.window_size // 2
		while current_window_index < data.shape[0] - args.window_size:
			window = normalize_window(data[current_window_index:current_window_index + args.window_size])
			windowed_data.append(window[..., np.newaxis])
			current_window_index += increment

		windowed_data = np.array(windowed_data)
		predictions, latent_spaces = model.predict(windowed_data)
		global_error = 0
		for pred, original in zip(predictions, windowed_data):
			global_error += np.sum((pred - original)**2)

		np.save(os.path.join(save_path, "complete_timeseries_input.npy"), data)
		np.save(os.path.join(save_path, "complete_timeseries_predicted.npy"), np.hstack(predictions))
		with open(os.path.join(save_path, "error.txt"), "w") as out_file:
			out_file.write(f"global_error:{global_error}\n")
	

def create_data_for_plots(args):
	path = os.path.join(args.save_predict, build_subpath_from_model_params(args))
	raw_save = os.path.join(args.save_plot_data, build_subpath_from_model_params(args))
	os.makedirs(raw_save, exist_ok=True)
	
	print("Searching for .npy files in {}".format(path))
	folders = sorted(glob.glob(os.path.join(path, "*.npy")))
	print("Found {} .mat files to create plots.".format(len(folders)))

	metrics = []

	for folder in folders:
		with open(os.path.join(folder, "error.txt"), "r") as error_file:
			error = 0
			for line in error_file:
				if "global_error" in line:
					error = float(line.split(":")[-1])

		metrics.append(error)

		print("{}: global:{:.4f}".format(folder, error))

	metrics = np.array(metrics)
	np.save(os.path.join(raw_save, "plot_data.npy"), metrics)

	folders = list(map(lambda x: path_leaf(x), folders))
	dates = [".".join(x.split(".")[:3][::-1]) for x in folders]
	ticks = [y + " " + ":".join(x.split(".")[-4:-1]) for x,y in zip(folders, dates)]
	with open(os.path.join(raw_save, "plot_data.ticks"), "wb") as dump_file:
		pickle.dump(ticks, dump_file)

	max_y = metrics.max()

	tick_skip = None

	if "Run1" in args.training_input_path:
		tick_skip = 33 
	elif "Run2" in args.training_input_path:
		tick_skip = 16 
	elif "Run3" in args.training_input_path:
		tick_skip = 33*3 

	################################
	######### Create plots #########
	################################

	# Overview plot (non moving average)
	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	matplotlib.rcParams.update({'font.size': 14})
	plt.title("Run " + str(args.run))
	plt.plot(metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(ticks) > 500:
		plt.xticks(np.arange(len(ticks))[::len(ticks)//500], ticks[::len(ticks)//500])
	else:
		plt.xticks(np.arange(len(ticks)), ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview.png"))
	plt.clf()

	plt.figure(figsize=(16, 9), dpi=100)
	plt.title("Run " + str(args.run))
	plt.plot(metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	plt.xticks(np.arange(len(ticks))[::tick_skip], ticks[::tick_skip])
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_presentation.png"))
	plt.clf()

	## Overview downsampled
	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title("Run " + str(args.run))
	plt.plot(metrics[::8])
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(ticks[::8]) > 500:
		plt.xticks(np.arange(len(ticks[::8]))[::len(ticks[::8])//500], ticks[::8][::len(ticks[::8])//500])
	else:
		plt.xticks(np.arange(len(ticks[::8])), ticks[::8])
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_downsampled.png"))
	plt.clf()

	## Overview moving average
	moving_average_window = 10

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title("Run " + str(args.run))
	plt.plot(ma_metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(ma_ticks) > 500:
		plt.xticks(np.arange(len(ma_ticks))[::len(ma_ticks)//500], ma_ticks[::len(ma_ticks)//500])
	else:
		plt.xticks(np.arange(len(ma_ticks)), ma_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_ma_10.png"))
	plt.clf()

	moving_average_window = 20

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title("Run " + str(args.run))
	plt.plot(ma_metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(ma_ticks) > 500:
		plt.xticks(np.arange(len(ma_ticks))[::len(ma_ticks)//500], ma_ticks[::len(ma_ticks)//500])
	else:
		plt.xticks(np.arange(len(ma_ticks)), ma_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_ma_20.png"))
	plt.clf()

	## Overview moving average downsampled

	moving_average_window = 10

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title("Run " + str(args.run))
	plt.plot(ma_metrics[::8])
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(ma_ticks[::8]) > 500:
		plt.xticks(np.arange(len(ma_ticks[::8]))[::len(ma_ticks[::8])//500], ma_ticks[::8][::len(ma_ticks[::8])//500])
	else:
		plt.xticks(np.arange(len(ma_ticks[::8])), ma_ticks[::8])
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_downsampled_ma_10.png"))
	plt.clf()

	moving_average_window = 20

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title("Run " + str(args.run))
	plt.plot(ma_metrics[::8])
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(ma_ticks[::8]) > 500:
		plt.xticks(np.arange(len(ma_ticks[::8]))[::len(ma_ticks[::8])//500], ma_ticks[::8][::len(ma_ticks[::8])//500])
	else:
		plt.xticks(np.arange(len(ma_ticks[::8])), ma_ticks[::8])
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	if max_y > 2000:
		plt.ylim(max(0, metrics.min() - 50), 1500)
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_downsampled_ma_20.png"))
	plt.clf()

	# ROI
	# plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	# plt.title("Run " + str(args.run))
	# plt.plot(roi_metrics)
	# plt.xlabel("Date")
	# plt.ylabel("Global error")
	# if len(roi_ticks) > 500:
	# 	plt.xticks(np.arange(len(roi_ticks))[::len(roi_ticks)//500], roi_ticks[::len(roi_ticks)//500])
	# else:
	# 	plt.xticks(np.arange(len(roi_ticks)), roi_ticks)
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roi.png"))
	# plt.clf()

	## ROI moving average
	# moving_average_window = 10

	# roi_ma_metrics = trailing_moving_average(roi_metrics, moving_average_window)
	# roi_ma_ticks = roi_ticks[moving_average_window - 1:]
	# assert(roi_ma_metrics.shape[0] == len(roi_ma_ticks))

	# plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	# plt.title("Run " + str(args.run))
	# plt.plot(roi_ma_metrics)
	# plt.xlabel("Date")
	# plt.ylabel("Global error")
	# if len(roi_ma_ticks) > 500:
	# 	plt.xticks(np.arange(len(roi_ma_ticks))[::len(roi_ma_ticks)//500], roi_ma_ticks[::len(roi_ma_ticks)//500])
	# else:
	# 	plt.xticks(np.arange(len(roi_ma_ticks)), roi_ma_ticks)
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roi_ma_10.png"))
	# plt.clf()

	# moving_average_window = 20

	# roi_ma_metrics = trailing_moving_average(roi_metrics, moving_average_window)
	# roi_ma_ticks = roi_ticks[moving_average_window - 1:]
	# assert(roi_ma_metrics.shape[0] == len(roi_ma_ticks))

	# plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	# plt.title("Run " + str(args.run))
	# plt.plot(roi_ma_metrics)
	# plt.xlabel("Date")
	# plt.ylabel("Global error")
	# if len(roi_ma_ticks) > 500:
	# 	plt.xticks(np.arange(len(roi_ma_ticks))[::len(roi_ma_ticks)//500], roi_ma_ticks[::len(roi_ma_ticks)//500])
	# else:
	# 	plt.xticks(np.arange(len(roi_ma_ticks)), roi_ma_ticks)
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roi_ma_20.png"))
	# plt.clf()

	# Create ROC and AUC inside of ROI

	# gt_list = load_txt_list(os.path.join("gt", "Run " + str(args.run) + ".txt"))
	# last_sample_in_gt = gt_list[-1]
	# first_sample_in_gt = gt_list[0]
	
	# joined = zip(metrics, ticks, folders)
	# filtered_joined = filter(lambda x: x[2] <= last_sample_in_gt, joined)

	# roc_metrics, roc_ticks, roc_folders = zip(*filtered_joined) # Unzip
	# roc_metrics = np.array(roc_metrics)

	# gt_begin_index = roc_folders.index(first_sample_in_gt)
	# gt_horizontal_line_y_value = roc_metrics[gt_begin_index]

	# gt_labels = []
	# for folder in roc_folders:
	# 	gt_labels.append(1 if folder in gt_list else 0)

	# assert(len(gt_labels) == len(roc_ticks))
	# assert(sum(gt_labels) == len(gt_list))

	# fpr, tpr, thresholds = roc_curve(gt_labels, roc_metrics)
	# roc_auc = roc_auc_score(gt_labels, roc_metrics)
	# if roc_auc < 0.5:
	# 	# Reverse the classifier
	# 	gt_labels = list(map(lambda x: 1 - x, gt_labels))
	# 	fpr, tpr, thresholds = roc_curve(gt_labels, roc_metrics)
	# 	roc_auc = roc_auc_score(gt_labels, roc_metrics)

	# plt.figure()
	# lw = 2
	# plt.plot(fpr, tpr, color='darkorange',
	# 		lw=lw, label="ROC curve (area = {:0.2f})".format(roc_auc))
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# num = 0
	# colors = iter(["r*", "g*", "y*"])
	# threshold_levels = []
	# for fpos, tpos, thresh in zip(fpr, tpr, thresholds):
	# 	dist = np.sum((np.array([0, 1]) - np.array([fpos, tpos]))**2)
	# 	threshold_levels.append((dist, fpos, tpos, thresh))

	# threshold_levels = sorted(threshold_levels, key=lambda x: x[0])

	# hline_thresholds = []
	# for i in range(3):
	# 	_, fpos, tpos, tresh = threshold_levels[i]
	# 	hline_thresholds.append(tresh)
	# 	plt.plot(fpos, tpos, next(colors))

	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic')
	# plt.legend(loc="lower right")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roc_annotated.png"))
	# plt.clf()

	# plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	# plt.title("Run " + str(args.run))
	# plt.plot(roc_metrics)
	# plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	# plt.hlines(hline_thresholds[0], 0, roc_metrics.shape[0], colors=["red"], linestyles="dashdot")
	# plt.hlines(hline_thresholds[1], 0, roc_metrics.shape[0], colors=["green"], linestyles="dashdot")
	# plt.hlines(hline_thresholds[2], 0, roc_metrics.shape[0], colors=["yellow"], linestyles="dashdot")
	# plt.xlabel("Date")
	# plt.ylabel("Global error")
	# if len(roc_ticks) > 500:
	# 	plt.xticks(np.arange(len(roc_ticks))[::len(roc_ticks)//500], roc_ticks[::len(roc_ticks)//500])
	# else:
	# 	plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roc_metric_annotated.png"))
	# plt.clf()

	# plt.figure()
	# lw = 2
	# plt.plot(fpr, tpr, color='darkorange',
	# 		lw=lw, label="ROC curve (area = {:0.2f})".format(roc_auc))
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic')
	# plt.legend(loc="lower right")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roc.png"))
	# plt.clf()

	# plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	# plt.title("Run " + str(args.run))
	# plt.plot(roc_metrics)
	# plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	# plt.xlabel("Date")
	# plt.ylabel("Global error")
	# if len(roc_ticks) > 500:
	# 	plt.xticks(np.arange(len(roc_ticks))[::len(roc_ticks)//500], roc_ticks[::len(roc_ticks)//500])
	# else:
	# 	plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roc_metric.png"))
	# plt.clf()

	# precision, recall, thresholds = precision_recall_curve(gt_labels, roc_metrics)
	# average_precision = average_precision_score(gt_labels, roc_metrics)
	# lw = 2
	# plt.figure()
	# plt.plot(recall, precision,
	# 		lw=lw, label="Average Precision = {:0.2f}".format(average_precision))
	# plt.xlim([0.0, 1.0])
	# plt.xlabel('Recall')
	# plt.ylabel('Precision')
	# plt.title('Precision-Recall curve')
	# plt.legend(loc="lower left")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "pr.png"))
	# plt.clf()

	# old_font_size = plt.rcParams.get("font.size")
	# plt.rcParams.update({'font.size': 22})
	# plt.figure(figsize=(16, 9), dpi=160)
	# plt.title("Run " + str(args.run))
	# plt.plot(roc_metrics)
	# plt.xlabel("Date")
	# plt.ylabel("Error")
	# tick_indices = (np.linspace(0, len(roc_ticks) - 1, 10)).astype(np.int)
	# plt.xticks(tick_indices, np.take(roc_ticks, tick_indices))
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roc_metric_for_presentation.png"))
	# plt.clf()
	# plt.rcParams.update({'font.size': old_font_size})

	# old_font_size = plt.rcParams.get("font.size")
	# plt.rcParams.update({'font.size': 22})
	# plt.figure(figsize=(16, 9), dpi=160)
	# plt.title("Run " + str(args.run))
	# plt.plot(roc_metrics)
	# plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	# plt.hlines(hline_thresholds[0], 0, roc_metrics.shape[0], colors=["red"], linestyles="dashdot")
	# plt.hlines(hline_thresholds[1], 0, roc_metrics.shape[0], colors=["green"], linestyles="dashdot")
	# plt.hlines(hline_thresholds[2], 0, roc_metrics.shape[0], colors=["yellow"], linestyles="dashdot")
	# plt.xlabel("Date")
	# plt.ylabel("Error")
	# tick_indices = (np.linspace(0, len(roc_ticks) - 1, 10)).astype(np.int)
	# plt.xticks(tick_indices, np.take(roc_ticks, tick_indices))
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
	# 		rotation_mode="anchor")
	# plt.tight_layout() 
	# plt.savefig(os.path.join(raw_save, "roc_metric_for_presentation_annotated.png"))
	# plt.clf()
	# plt.rcParams.update({'font.size': old_font_size})

def main(args):
	print("nsddjfksdk")
	if args.use_reduced_model:
		args.filter_num = 8 // args.filter_ratio
	else:
		args.filter_num = 64 // args.filter_ratio

	print("Building model")
	model = None
	if args.architecture == "CNN":
		model_parameters = {"window_size": args.window_size,
							"filter_num": args.filter_num,
							"filter_size": args.filter_size,
							"num_conv": args.num_conv,
							"reduced_model": args.use_reduced_model}
		
		model = build_cnn_model(model_parameters)
	
	enc_model = None
	dec_model = None
	if not args.skip_training:
		print("Starting training")
		print("Loading training data")
		train(model, args)

	print("Loading test data")
	if not args.skip_prediction:
		if args.skip_training:
			if args.architecture == "CNN":
				if args.model_weights:
					model.load_weights(args.model_weights)
				else:
					model_save_path = os.path.join(args.save_model, build_subpath_from_model_params(args), "cnn_model.h5" if args.architecture == "CNN" else "lstm_model.h5")
					model.load_weights(model_save_path)

		predict(model, args)

	print("Creating plots")
	create_data_for_plots(args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-w", "--workers", type=int, default=0, help="Number of additional worker threads. Default = 0")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default = 100")
	parser.add_argument("-ef", "--epochs_per_file", type=int, default=5, help="Number of epochs per .mat file. Default = 5")
	parser.add_argument("-sm", "--save_model", type=str, default="trained_models", help="Save path for trained model. Default = 'trained_models'")
	parser.add_argument("-sp", "--save_predict", type=str, default="model_predictions", help="Save path for model predictions. Default = 'model_predictions'")
	parser.add_argument("-sd", "--save_plot_data", type=str, default="plot_data", help="Save path for plot data. Default = 'plot_data'")
	parser.add_argument("-n", "--name", type=str, required=True, help="Unique name of the run.")

	parser.add_argument("-r", "--run", type=int, choices=[1, 2, 3], required=True, help="Which run are we looking at? Possible values: 1, 2 or 3.")
	parser.add_argument("-l", "--lager", type=str, required=True)
	parser.add_argument("-a", "--architecture", type=str, choices=["CNN"], default="CNN", help="Type of network. Must be 'CNN'.")

	parser.add_argument("--skip_training", action="store_true", help="Skips the training step.")
	parser.add_argument("--use_reduced_model", action="store_true", help="Builds a model suitable to use window size 8.")
	parser.add_argument("-mw", "--model_weights", type=str, help="Path to model weights. Only used if --skip_trainig is set.")
	parser.add_argument("--normalize_then_split", action="store_true", help="Normalize the data first and then split into chunks of size window_size.")

	parser.add_argument("--skip_prediction", action="store_true", help="Skips the prediction step.")

	## data loader specific
	parser.add_argument("-ti", "--training_input_path", type=str, required=True, help="Path to .mat training files.")
	# parser.add_argument("-pi", "--predict_input_path", type=str, required=True, help="Path to .mat files to predict.")
	parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size. Default = 1024")
	parser.add_argument("-ws", "--window_size", type=int, default=128, help="Window size of loaded data. Default = 128")
	parser.add_argument("-sk", "--skip_train_files", type=int, default=0, help="Number of train files to skip. Default = 0")
	# parser.add_argument("--sensor", type=int, default=0, help="What sensor data to use. Acceptable values range: [0,3]. Default = 0")
	## cnn specific
	parser.add_argument("-nc", "--num_conv", type=int, default=1, help="Number of convolution in each downward/upward layer of the network. Default = 1")
	parser.add_argument("-f", "--filter_num", type=int, default=None, help="Number of filters per 1d-conv layer")
	parser.add_argument("-fs", "--filter_size", type=int, default=3, help="Filter size of a convolution. Set to Size x Size and must be odd. Default = 3")
	parser.add_argument("-fr", "--filter_ratio", type=int, help="Percentage of window size representing the latent space size. Example: Window Size=128, Filter ratio=2 --> 128/2=64 Latent Space Dim and 64/2=32 --> Number of 1d-conv filters.")

	args = parser.parse_args()

	main(args)
