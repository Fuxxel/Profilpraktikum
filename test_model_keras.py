from models.keras.enc_dec import build_enc_dec_model_test
from models.keras.cnn import build_cnn_model_test
from dataloader import Test_DataLoader

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
	model_parameters = {"latent_dim": args.latent_dim, 
						"window_size": args.window_size,
						"filter_num": args.filter_num}
	model_build_fns = {"cnn": build_cnn_model_test,
					   "enc_dec": build_enc_dec_model_test}
	data_loader_classes = {"cnn": Test_DataLoader,
					   	   "enc_dec": Test_DataLoader}
	save_file_names = {"cnn": "cnn_model.h5",
					   "enc_dec": "enc_dec_model.h5"}

	data_loader = data_loader_classes[args.architecture](data_folder=args.input_path, 
														 batch_size=args.batch_size, 
														 sample_length=args.window_size)

	if args.architecture == "enc_dec":
		complete_model, encoder_model, decoder_model = model_build_fns[args.architecture](model_parameters, args.h5_file)
		for current_file_index in range(data_loader.num_files()):
			print("\r{}/{}".format(current_file_index, data_loader.num_files(), end=""))
			save_path = os.path.join(args.save, data_loader.current_file())
			os.makedirs(save_path, exist_ok=True)

			complete_timeseries_input = []
			complete_timeseries_predicted = []

			for current_batch_index, batch in enumerate(data_loader):
				# Encode all inputs in batch into intial state vectors
				state_values = encoder_model.predict(batch)

				previous_outputs = np.full((args.batch_size, 1, 1), 0) # Initial value zero 
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

					# plt.plot(gt_sample, color="blue", label="Input")
					# plt.plot(predicted_sample, color="red", label="Predicted")

					# np.save(os.path.join(save_path, "{:06d}_{:06d}_input.npy".format(current_batch_index, batch_index)), gt_sample)
					# np.save(os.path.join(save_path, "{:06d}_{:06d}_predicted.npy".format(current_batch_index, batch_index)), predicted_sample)
					# plt.savefig(os.path.join(save_path, "{:06d}_{:06d}.png".format(current_batch_index, batch_index)))
					# plt.clf()

			complete_timeseries_input = np.array(complete_timeseries_input)
			complete_timeseries_predicted = np.array(complete_timeseries_predicted)

			shape = complete_timeseries_input.shape
			complete_timeseries_input = np.reshape(complete_timeseries_input, shape[0] * shape[1])
			complete_timeseries_predicted = np.reshape(complete_timeseries_predicted, shape[0] * shape[1])

			np.save(os.path.join(save_path, "complete_timeseries_input.npy"), complete_timeseries_input)
			np.save(os.path.join(save_path, "complete_timeseries_predicted.npy"), complete_timeseries_predicted)

			data_loader.on_epoch_end()
	elif args.architecture == "cnn":
		complete_model = model_build_fns[args.architecture](model_parameters, args.h5_file)
		for current_file_index in range(data_loader.num_files()):
			print("\r{}/{}".format(current_file_index, data_loader.num_files(), end=""))
			save_path = os.path.join(args.save, data_loader.current_file())
			os.makedirs(save_path, exist_ok=True)

			complete_timeseries_input = []
			complete_timeseries_predicted = []

			for current_batch_index, batch in enumerate(data_loader):

				predicted_sequence = complete_model.predict(batch)
				
				for batch_index in range(predicted_sequence.shape[0]):
					gt_sample = np.squeeze(batch[batch_index], -1)
					predicted_sample = np.squeeze(predicted_sequence[batch_index], -1)

					complete_timeseries_input.append(gt_sample)
					complete_timeseries_predicted.append(predicted_sample)

					# plt.plot(gt_sample, color="blue", label="Input")
					# plt.plot(predicted_sample, color="red", label="Predicted")

					# np.save(os.path.join(save_path, "{:06d}_{:06d}_input.npy".format(current_batch_index, batch_index)), gt_sample)
					# np.save(os.path.join(save_path, "{:06d}_{:06d}_predicted.npy".format(current_batch_index, batch_index)), predicted_sample)
					# plt.savefig(os.path.join(save_path, "{:06d}_{:06d}.png".format(current_batch_index, batch_index)))
					# plt.clf()

			complete_timeseries_input = np.array(complete_timeseries_input)
			complete_timeseries_predicted = np.array(complete_timeseries_predicted)

			shape = complete_timeseries_input.shape
			complete_timeseries_input = np.reshape(complete_timeseries_input, shape[0] * shape[1])
			complete_timeseries_predicted = np.reshape(complete_timeseries_predicted, shape[0] * shape[1])

			np.save(os.path.join(save_path, "complete_timeseries_input.npy"), complete_timeseries_input)
			np.save(os.path.join(save_path, "complete_timeseries_predicted.npy"), complete_timeseries_predicted)

			data_loader.on_epoch_end()
	print()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-a", "--architecture", type=str, choices=["cnn", "enc_dec"], required=True, help="Network architecture. Possible options: 'cnn', 'enc_dec'")
	parser.add_argument("-w", "--workers", type=int, default=0, help="Number of additional worker threads. Default = 0")
	parser.add_argument("-s", "--save", type=str, default="test_results", help="Save path for test results. Default = 'test_results'")
	parser.add_argument("-h5", "--h5_file", type=str, required=True, help="Path to .h5 model weights file.")

	## data loader specific
	parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to .mat data files for testing.")
	parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size. Default = 50")
	parser.add_argument("-ws", "--window_size", type=int, default=1024, help="Window size of loaded data")

	## enc_dec specific
	parser.add_argument("-d", "--latent_dim", type=int, default=100, help="Latent dim. Default = 100")

	## cnn specific
	parser.add_argument("-f", "--filter_num", type=int, default=16, help="Number of filters per 1d-conv layer. Default = 16")

	
	args = parser.parse_args()

	main(args)
