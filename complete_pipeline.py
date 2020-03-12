from models.keras.cnn import build_cnn_model
from models.keras.enc_dec import build_enc_dec_model, build_enc_dec_model_test
from dataloader import CNN_DataLoader, CNN_Test_DataLoader, LSTM_DataLoader, LSTM_Test_DataLoader

import argparse
import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

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
	# return os.path.join(args.name, "ws{}".format(args.window_size), "nc{}".format(args.num_conv), "fs{}".format(args.filter_size), "skip_first_{}".format(args.skip_train_files), "lat_{}".format(args.filter_ratio))
	# return os.path.join(args.name, "reduced_model" if args.use_reduced_model else "full_model", "sensor_{}".format(args.sensor), "ws{}".format(args.window_size), "nc{}".format(args.num_conv), "fs{}".format(args.filter_size), "skip_first_{}".format(args.skip_train_files), "lat_{}".format(args.filter_ratio))
	return os.path.join(args.name, args.architecture, "reduced_model" if args.use_reduced_model else "full_model", "normalize_then_split" if args.normalize_then_split else "", "sensor_{}".format(args.sensor), "ws{}".format(args.window_size), "nc{}".format(args.num_conv), "fs{}".format(args.filter_size), "skip_first_{}".format(args.skip_train_files), "lat_{}".format(args.filter_ratio))

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

def train(model, data_loader, args):
	print("##########################")
	print("Starting training")
	print("##########################")
	save_file_name = "cnn_model.h5" if args.architecture == "CNN" else "lstm_model.h5"

	os.makedirs(os.path.join(args.save_model, build_subpath_from_model_params(args)), exist_ok=True)
	model_save_path = os.path.join(args.save_model, build_subpath_from_model_params(args), save_file_name)

	status_path = os.path.join(args.save_model, build_subpath_from_model_params(args), "status.txt")

	status = {"last_epoch": 0}
	if os.path.exists(status_path):
		status = read_train_status_file(status_path)

	start_at_epoch = status["last_epoch"]

	if start_at_epoch > 0: # Advance the dataloader to desired starting epoch
		for _ in range(start_at_epoch // args.epochs_per_file):
			data_loader.on_epoch_end()

	for current_epoch in range(start_at_epoch, args.epochs, args.epochs_per_file):
		model.fit_generator(data_loader,
							use_multiprocessing=(args.workers != 0), 
							workers=args.workers,
							epochs=current_epoch + args.epochs_per_file,
							initial_epoch=current_epoch)
		data_loader.on_epoch_end()
		model.save(model_save_path)
		status["last_epoch"] = current_epoch
		write_train_status_file(status, status_path)

	print("Saving model to: {}".format(model_save_path))
	model.save(model_save_path)
	
	return model

def predict(model, data_loader, args):
	print("##########################")
	print("Starting prediction")
	print("##########################")
	encoder_model = None 
	decoder_model = None
	if args.architecture == "LSTM":
		model, encoder_model, decoder_model = model

	for current_file_index in range(data_loader.num_files()):
		print("\r{}/{}".format(current_file_index, data_loader.num_files(), end=""))
		save_path = os.path.join(args.save_predict, build_subpath_from_model_params(args), data_loader.current_file())

		if args.skip_training and \
		   os.path.exists(save_path) and \
		   os.path.exists(os.path.join(save_path, "complete_timeseries_input.npy")) and \
		   os.path.exists(os.path.join(save_path, "complete_timeseries_predicted.npy")) and \
		   os.path.exists(os.path.join(save_path, "latent_spaces.npy")):
		   print("Prediction data for {} already exists. Skipping.".format(data_loader.current_file()))
		   data_loader.on_epoch_end()
		   continue

		data_loader.on_epoch_start()
		os.makedirs(save_path, exist_ok=True)

		complete_timeseries_input = []
		complete_timeseries_predicted = []

		complete_latent_spaces = []

		if args.architecture == "CNN":
			for current_batch_index, batch in enumerate(data_loader):
				print("\r{}/{}".format(current_batch_index + 1, len(data_loader)), end="")
				predicted_sequence, latent_spaces = model.predict_on_batch(batch)
				complete_latent_spaces.append(latent_spaces)
				
				for batch_index in range(predicted_sequence.shape[0]):
					gt_sample = np.squeeze(batch[batch_index], -1)
					predicted_sample = np.squeeze(predicted_sequence[batch_index], -1)

					complete_timeseries_input.append(gt_sample)
					complete_timeseries_predicted.append(predicted_sample)
			print("")

			complete_timeseries_input = np.array(complete_timeseries_input)
			complete_timeseries_predicted = np.array(complete_timeseries_predicted)
			complete_latent_spaces = np.asarray(complete_latent_spaces)

			shape = complete_timeseries_input.shape
			complete_timeseries_input = np.reshape(complete_timeseries_input, shape[0] * shape[1])
			complete_timeseries_predicted = np.reshape(complete_timeseries_predicted, shape[0] * shape[1])
			
			np.save(os.path.join(save_path, "complete_timeseries_input.npy"), complete_timeseries_input)
			np.save(os.path.join(save_path, "complete_timeseries_predicted.npy"), complete_timeseries_predicted)
			np.save(os.path.join(save_path, "latent_spaces.npy"), complete_latent_spaces)

			data_loader.on_epoch_end()
		else:
			for current_batch_index, batch in enumerate(data_loader):
				# Encode all inputs in batch into intial state vectors
				state_values = encoder_model.predict_on_batch(batch)

				complete_latent_spaces.append(state_values[0])

				previous_outputs = np.full((args.batch_size, 1, 1), -1) # Initial value minus one 
				predicted_sequence = np.zeros_like(batch)
				for current_predict_index in range(args.window_size):
					output, h, c = decoder_model.predict([previous_outputs] + state_values)

					# Model predicts backwards!
					predicted_sequence[:, args.window_size - current_predict_index - 1] = np.squeeze(output, -1)

					previous_outputs = output
					state_values = [h, c]

				for batch_index in range(predicted_sequence.shape[0]):
					gt_sample = np.squeeze(batch[batch_index], -1)
					predicted_sample = np.squeeze(predicted_sequence[batch_index], -1)

					complete_timeseries_input.append(gt_sample)
					complete_timeseries_predicted.append(predicted_sample)

			complete_timeseries_input = np.array(complete_timeseries_input)
			complete_timeseries_predicted = np.array(complete_timeseries_predicted)

			shape = complete_timeseries_input.shape
			complete_timeseries_input = np.reshape(complete_timeseries_input, shape[0] * shape[1])
			complete_timeseries_predicted = np.reshape(complete_timeseries_predicted, shape[0] * shape[1])

			np.save(os.path.join(save_path, "complete_timeseries_input.npy"), complete_timeseries_input)
			np.save(os.path.join(save_path, "complete_timeseries_predicted.npy"), complete_timeseries_predicted)
			np.save(os.path.join(save_path, "latent_spaces.npy"), complete_latent_spaces)

			data_loader.on_epoch_end()

def create_data_for_plots(args):
	path = os.path.join(args.save_predict, build_subpath_from_model_params(args))
	raw_save = os.path.join(args.save_plot_data, build_subpath_from_model_params(args))
	os.makedirs(raw_save, exist_ok=True)
	
	print("Searching for .mat files in {}".format(path))
	folders = os.listdir(path)
	folders = filter(lambda x: x.endswith(".mat"), folders)
	folders = sorted(folders)
	print("Found {} .mat files to create plots.".format(len(folders)))

	metrics = []

	for folder in folders:
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

	metrics = np.array(metrics)
	np.save(os.path.join(raw_save, "plot_data.npy"), metrics)

	dates = [".".join(x.split("_")[1:3][::-1]) for x in folders]
	ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(folders, dates)]
	with open(os.path.join(raw_save, "plot_data.ticks"), "wb") as dump_file:
		pickle.dump(ticks, dump_file)

	################################
	######### Create plots #########
	################################

	metrics = metrics[:, 2]

	# Create ROI metrics and ticks
	roi_list = load_txt_list(os.path.join("roi", args.lager + ".txt"))
	joined = zip(metrics, ticks, folders)
	filtered_joined = filter(lambda x: x[2] in roi_list, joined)

	roi_metrics, roi_ticks, roi_folders = zip(*filtered_joined) # Unzip
	roi_metrics = np.array(roi_metrics)

	# Overview plot (non moving average)
	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
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
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview.png"))
	plt.clf()

	## Overview downsampled
	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
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
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_downsampled.png"))
	plt.clf()

	## Overview moving average
	moving_average_window = 10

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
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
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_ma_10.png"))
	plt.clf()

	moving_average_window = 20

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
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
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_ma_20.png"))
	plt.clf()

	## Overview moving average downsampled

	moving_average_window = 10

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
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
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_downsampled_ma_10.png"))
	plt.clf()

	moving_average_window = 20

	ma_metrics = trailing_moving_average(metrics, moving_average_window)
	ma_ticks = ticks[moving_average_window - 1:]
	assert(ma_metrics.shape[0] == len(ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
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
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "overview_downsampled_ma_20.png"))
	plt.clf()

	# ROI
	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
	plt.plot(roi_metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(roi_ticks) > 500:
		plt.xticks(np.arange(len(roi_ticks))[::len(roi_ticks)//500], roi_ticks[::len(roi_ticks)//500])
	else:
		plt.xticks(np.arange(len(roi_ticks)), roi_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "roi.png"))
	plt.clf()

	## ROI moving average
	moving_average_window = 10

	roi_ma_metrics = trailing_moving_average(roi_metrics, moving_average_window)
	roi_ma_ticks = roi_ticks[moving_average_window - 1:]
	assert(roi_ma_metrics.shape[0] == len(roi_ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
	plt.plot(roi_ma_metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(roi_ma_ticks) > 500:
		plt.xticks(np.arange(len(roi_ma_ticks))[::len(roi_ma_ticks)//500], roi_ma_ticks[::len(roi_ma_ticks)//500])
	else:
		plt.xticks(np.arange(len(roi_ma_ticks)), roi_ma_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "roi_ma_10.png"))
	plt.clf()

	moving_average_window = 20

	roi_ma_metrics = trailing_moving_average(roi_metrics, moving_average_window)
	roi_ma_ticks = roi_ticks[moving_average_window - 1:]
	assert(roi_ma_metrics.shape[0] == len(roi_ma_ticks))

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
	plt.plot(roi_ma_metrics)
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(roi_ma_ticks) > 500:
		plt.xticks(np.arange(len(roi_ma_ticks))[::len(roi_ma_ticks)//500], roi_ma_ticks[::len(roi_ma_ticks)//500])
	else:
		plt.xticks(np.arange(len(roi_ma_ticks)), roi_ma_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "roi_ma_20.png"))
	plt.clf()

	# Create ROC and AUC inside of ROI
	gt_list = load_txt_list(os.path.join("gt", args.lager + ".txt"))
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
	plt.savefig(os.path.join(raw_save, "roc_annotated.png"))
	plt.clf()

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
	plt.plot(roc_metrics)
	plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	plt.hlines(hline_thresholds[0], 0, roc_metrics.shape[0], colors=["red"], linestyles="dashdot")
	plt.hlines(hline_thresholds[1], 0, roc_metrics.shape[0], colors=["green"], linestyles="dashdot")
	plt.hlines(hline_thresholds[2], 0, roc_metrics.shape[0], colors=["yellow"], linestyles="dashdot")
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(roc_ticks) > 500:
		plt.xticks(np.arange(len(roc_ticks))[::len(roc_ticks)//500], roc_ticks[::len(roc_ticks)//500])
	else:
		plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "roc_metric_annotated.png"))
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
	plt.savefig(os.path.join(raw_save, "roc.png"))
	plt.clf()

	plt.figure(figsize=(11.27*12, 7.04), dpi=227)
	plt.title(args.lager)
	plt.plot(roc_metrics)
	plt.hlines(gt_horizontal_line_y_value, 0, roc_metrics.shape[0])
	plt.xlabel("Date")
	plt.ylabel("Global error")
	if len(roc_ticks) > 500:
		plt.xticks(np.arange(len(roc_ticks))[::len(roc_ticks)//500], roc_ticks[::len(roc_ticks)//500])
	else:
		plt.xticks(np.arange(len(roc_ticks)), roc_ticks)
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "roc_metric.png"))
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
	plt.savefig(os.path.join(raw_save, "pr.png"))
	plt.clf()

	old_font_size = plt.rcParams.get("font.size")
	plt.rcParams.update({'font.size': 22})
	plt.figure(figsize=(16, 9), dpi=80)
	plt.title(args.lager)
	plt.plot(roc_metrics)
	plt.xlabel("Date")
	plt.ylabel("Error")
	tick_indices = (np.linspace(0, len(roc_ticks) - 1, 10)).astype(np.int)
	plt.xticks(tick_indices, np.take(roc_ticks, tick_indices))
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
			rotation_mode="anchor")
	plt.tight_layout() 
	plt.savefig(os.path.join(raw_save, "roc_metric_for_presentation.png"))
	plt.clf()
	plt.rcParams.update({'font.size': old_font_size})

def main(args):
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
	else:
		model_parameters = {"latent_dim": args.window_size // args.filter_ratio}
		
		model = build_enc_dec_model(model_parameters)
		model_save_path = os.path.join(args.save_model, build_subpath_from_model_params(args), "cnn_model.h5" if args.architecture == "CNN" else "lstm_model.h5")
		if os.path.exists(model_save_path):
			model.load_weights(model_save_path)


	enc_model = None
	dec_model = None
	if not args.skip_training:
		print("Starting training")
		print("Loading training data")
		training_data_loader = None
		if args.architecture == "CNN":
			training_data_loader = CNN_DataLoader(data_folder=args.training_input_path, 
												  batch_size=args.batch_size, 
												  sample_length=args.window_size,
												  sensor=args.sensor,
												  skip_files=args.skip_train_files,
												  normalize_then_split=args.normalize_then_split)
		else:
			training_data_loader = LSTM_DataLoader(data_folder=args.training_input_path, 
												   batch_size=args.batch_size, 
												   sample_length=args.window_size,
												   sensor=args.sensor,
												   skip_files=args.skip_train_files,
												   normalize_then_split=args.normalize_then_split)
	
		train(model, training_data_loader, args)

	print("Loading test data")
	if args.architecture == "CNN":
		predict_data_loader = CNN_Test_DataLoader(data_folder=args.predict_input_path, 
												 batch_size=args.batch_size, 
												 sensor=args.sensor,
												 sample_length=args.window_size,
												 normalize_then_split=args.normalize_then_split)
	else:
		predict_data_loader = LSTM_Test_DataLoader(data_folder=args.predict_input_path, 
												   batch_size=args.batch_size, 
												   sensor=args.sensor,
												   sample_length=args.window_size,
												   normalize_then_split=args.normalize_then_split)
	
	if not args.skip_prediction:
		if args.skip_training:
			if args.architecture == "CNN":
				if args.model_weights:
					model.load_weights(args.model_weights)
				else:
					model_save_path = os.path.join(args.save_model, build_subpath_from_model_params(args), "cnn_model.h5" if args.architecture == "CNN" else "lstm_model.h5")
					model.load_weights(model_save_path)
			else:
				if args.model_weights:
					model = build_enc_dec_model_test(model_parameters, args.model_weights)
				else:
					model_save_path = os.path.join(args.save_model, build_subpath_from_model_params(args), "cnn_model.h5" if args.architecture == "CNN" else "lstm_model.h5")
					model = build_enc_dec_model_test(model_parameters, model_save_path)

		if args.architecture == "LSTM":
			model, enc_model, dec_model = build_enc_dec_model_test(model_parameters, os.path.join(args.save_model, build_subpath_from_model_params(args), "cnn_model.h5" if args.architecture == "CNN" else "lstm_model.h5"))
			predict((model, enc_model, dec_model), predict_data_loader, args)
		else:
			predict(model, predict_data_loader, args)

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

	parser.add_argument("-l", "--lager", type=str, choices=["Lager4", "Lager5"], required=True, help="Which Lager are we looking at? Possible values: 'Lager4' and 'Lager5'.")
	parser.add_argument("-a", "--architecture", type=str, choices=["CNN", "LSTM"], default="CNN", help="Type of network. Defaults to 'CNN'.")

	parser.add_argument("--skip_training", action="store_true", help="Skips the training step.")
	parser.add_argument("--use_reduced_model", action="store_true", help="Builds a model suitable to use window size 8.")
	parser.add_argument("-mw", "--model_weights", type=str, help="Path to model weights. Only used if --skip_trainig is set.")
	parser.add_argument("--normalize_then_split", action="store_true", help="Normalize the data first and then split into chunks of size window_size.")

	parser.add_argument("--skip_prediction", action="store_true", help="Skips the prediction step.")

	## data loader specific
	parser.add_argument("-ti", "--training_input_path", type=str, required=True, help="Path to .mat training files.")
	parser.add_argument("-pi", "--predict_input_path", type=str, required=True, help="Path to .mat files to predict.")
	parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size. Default = 1024")
	parser.add_argument("-ws", "--window_size", type=int, default=128, help="Window size of loaded data. Default = 128")
	parser.add_argument("-sk", "--skip_train_files", type=int, default=0, help="Number of train files to skip. Default = 0")
	parser.add_argument("--sensor", type=int, default=0, help="What sensor data to use. Acceptable values range: [0,3]. Default = 0")
	## cnn specific
	parser.add_argument("-nc", "--num_conv", type=int, default=1, help="Number of convolution in each downward/upward layer of the network. Default = 1")
	parser.add_argument("-f", "--filter_num", type=int, default=None, help="Number of filters per 1d-conv layer")
	parser.add_argument("-fs", "--filter_size", type=int, default=3, help="Filter size of a convolution. Set to Size x Size and must be odd. Default = 3")
	parser.add_argument("-fr", "--filter_ratio", type=int, help="Percentage of window size representing the latent space size. Example: Window Size=128, Filter ratio=2 --> 128/2=64 Latent Space Dim and 64/2=32 --> Number of 1d-conv filters.")

	args = parser.parse_args()

	main(args)
