from models.enc_dec import build_enc_dec_model
from dataloader import DataLoader

import argparse

def main(args):
	model_parameters = {"latent_dim": args.latent_dim}
	enc_dec_model = build_enc_dec_model(model_parameters)

	data_loader = DataLoader(data_folder="Sample Data", batch_size=args.batch_size)

	enc_dec_model.fit_generator(data_loader,
							    use_multiprocessing=(args.workers != 0), 
								workers=args.workers,
								epochs=args.epochs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size. Default = 50")
	parser.add_argument("-d", "--latent_dim", type=int, default=1024, help="Latent dim. Default = 1024")
	parser.add_argument("-w", "--workers", type=int, default=0, help="Number of additional worker threads. Default = 0")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default = 100")
	args = parser.parse_args()

	main(args)
