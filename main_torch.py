import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import DataLoader
from tqdm import tqdm

import numpy as np
from random import shuffle
import argparse

sequence_length = 1024

def to_cpu_tensor(o):
	return torch.from_numpy(o).float()

def to_gpu_tensor(o):
	return torch.from_numpy(o).float().cuda()

def main(args):
	gpu_available = torch.cuda.is_available()

	model = nn.Transformer(d_model=1, nhead=1)
	if gpu_available:
		model.cuda()

	loss = nn.MSELoss()
	optimizer = optim.Adam(model.parameters())

	to_tensor = to_gpu_tensor if gpu_available else to_cpu_tensor

	data_loader = DataLoader(data_folder="Sample Data", batch_size=args.batch_size)

	batch_indices = list(range(len(data_loader))) 
	shuffle(batch_indices)

	# Build target masks for each batch
	# 0 -inf -inf -inf ...
	# 0    0 -inf -inf ...
	# 0    0    0 -inf ...
	mask = np.triu(np.full((sequence_length, sequence_length), float("-inf")), 1)
	mask = to_tensor(mask)

	for epoch in range(args.epochs):
		for batch_index in tqdm(batch_indices, desc="Epoch {}".format(epoch + 1)):
			optimizer.zero_grad()

			input_batch, output_batch = data_loader[batch_index]
			# Expected format: (Sequence length x Batch size x Feature dim)
			encoder_inputs = to_tensor(np.swapaxes(input_batch["encoder_inputs"], 0, 1))
			decoder_inputs = to_tensor(np.swapaxes(input_batch["decoder_inputs"], 0, 1))

			expected_output = to_tensor(output_batch) # In (Batch size x Sequence length x Feature dim)

			model_output = model(encoder_inputs, decoder_inputs, tgt_mask=mask).transpose_(0, 1) # In (Batch size x Sequence length x Feature dim)
			error = loss(expected_output, model_output)
			error.backward()
			optimizer.step()

			print("\r{}".format(error.item()), end="")
	print()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size. Default = 50")
	# parser.add_argument("-d", "--latent_dim", type=int, default=1024, help="Latent dim. Default = 1024")
	# parser.add_argument("-w", "--workers", type=int, default=0, help="Number of additional worker threads. Default = 0")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default = 100")
	args = parser.parse_args()

	main(args)
