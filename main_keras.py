from models.keras.enc_dec import build_enc_dec_model
from models.keras.cnn import build_cnn_model
from dataloader import Enc_Dec_DataLoader, CNN_DataLoader

import argparse
import os

def main(args):
	model_parameters = {"latent_dim": args.latent_dim, 
						"window_size": args.window_size,
						"filter_num": args.filter_num}
	model_build_fns = {"cnn": build_cnn_model,
					   "enc_dec": build_enc_dec_model}
	data_loader_classes = {"cnn": CNN_DataLoader,
					   	   "enc_dec": Enc_Dec_DataLoader}
	save_file_names = {"cnn": "cnn_model.h5",
					   "enc_dec": "enc_dec_model.h5"}

	data_loader = data_loader_classes[args.architecture](data_folder=args.input_path, 
														 batch_size=args.batch_size, 
														 sample_length=args.window_size)
	
	model = model_build_fns[args.architecture](model_parameters)

	for current_epoch in range(0, args.epochs, args.epochs_per_file):
		model.fit_generator(data_loader,
							use_multiprocessing=(args.workers != 0), 
							workers=args.workers,
							epochs=current_epoch + args.epochs_per_file,
							initial_epoch=current_epoch)
		data_loader.on_epoch_end()

	os.makedirs(args.save, exist_ok=True)
	model.save(os.path.join(args.save, save_file_names[args.architecture]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-a", "--architecture", type=str, choices=["cnn", "enc_dec"], required=True, help="Network architecture. Possible options: 'cnn', 'enc_dec'")
	parser.add_argument("-w", "--workers", type=int, default=0, help="Number of additional worker threads. Default = 0")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default = 100")
	parser.add_argument("-ef", "--epochs_per_file", type=int, default=5, help="Number of epochs per .mat file. Default = 5")
	parser.add_argument("-s", "--save", type=str, default="trained_models", help="Save path for trained model. Default = 'trained_models'")
	
	## data loader specific
	parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to .mat training files.")
	parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size. Default = 50")
	parser.add_argument("-ws", "--window_size", type=int, default=1024, help="Window size of loaded data")

	## enc_dec specific
	parser.add_argument("-d", "--latent_dim", type=int, default=100, help="Latent dim. Default = 100")

	## cnn specific
	parser.add_argument("-f", "--filter_num", type=int, default=16, help="Number of filters per 1d-conv layer. Default = 16")

	
	args = parser.parse_args()

	main(args)
