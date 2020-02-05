from models.keras.cnn import build_cnn_model_test
from dataloader import Test_DataLoader

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
	model_parameters = {"window_size": args.window_size,
						"filter_num": args.filter_num}

	data_loader = Test_DataLoader(data_folder=args.input_path, 
								  batch_size=args.batch_size, 
								  sample_length=args.window_size)

	complete_model = build_cnn_model_test(model_parameters, args.h5_file)
	for current_file_index in range(data_loader.num_files()):
		print("\r{}/{}".format(current_file_index, data_loader.num_files(), end=""))
		save_path = os.path.join(args.save, data_loader.current_file())
		os.makedirs(save_path, exist_ok=True)

		complete_latent_spaces = []
		for current_batch_index, batch in enumerate(data_loader):
			_, latent_spaces = complete_model.predict(batch)
			complete_latent_spaces.append(latent_spaces)

		complete_latent_spaces = np.array(complete_latent_spaces)
		np.save(os.path.join(save_path, "latent_spaces.npy"), complete_latent_spaces)

		data_loader.on_epoch_end()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to .mat files.")
	parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size. Default = 1024")
	
	parser.add_argument("-s", "--save", type=str, required=True, help="Save path for latent spaces.")
	parser.add_argument("-h5", "--h5_file", type=str, required=True, help="Path to model weights.")
	parser.add_argument("-ws", "--window_size", type=int, required=True, help="Window size of loaded data.")
	parser.add_argument("-f", "--filter_num", type=int, required=True, help="Number of filters per 1d-conv layer.")
	
	args = parser.parse_args()

	main(args)