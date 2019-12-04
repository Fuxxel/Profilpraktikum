import tensorflow as tf

from scipy.io import loadmat
import numpy as np
import os
import random
import math

class Test_DataLoader(tf.keras.utils.Sequence):
	def __init__(self, data_folder, batch_size, sample_length=1024):
		self.data_folder = data_folder
		self.skip_ahead = sample_length
		self.batch_size = batch_size
		self.sample_length = sample_length

		self.__load_filenames()

		self.current_file_index = 0
		self.__load_next_file_into_memory()

		self.__create_batch_indices()

	def __create_batch_indices(self):
		self.batch_indices = list(range(0, self.current_timeseries.shape[0] - self.sample_length, self.skip_ahead))

	def num_files(self):
		return len(self.files)

	def current_file(self):
		return self.filenames[self.current_file_index]

	def on_epoch_end(self):
		# Load new file from datafolder into memory and train.
		# New file in consecutive order of file array.
		self.current_file_index = (self.current_file_index + 1) % len(self.files)

		self.__load_next_file_into_memory()

		self.__create_batch_indices()

	def __normalize_sample(self, sample):
		max = np.max(sample)
		min = np.min(sample)
		return (sample - min) / (max - min)

	def __load_next_file_into_memory(self):
		file_to_load = self.files[self.current_file_index]
		print("Processing: {}".format(file_to_load))

		self.current_timeseries = loadmat(file_to_load)["Data"][..., 0, np.newaxis] # Take zero'th timeseries

	def __len__(self):
		return math.floor(len(self.batch_indices) // self.batch_size)

	def __getitem__(self, idx):
		batch_input = []
		for i in range(idx, idx + self.batch_size):
			current_index = self.batch_indices[i]
			sample = self.current_timeseries[current_index:current_index + self.sample_length]
			sample = self.__normalize_sample(sample)

			batch_input.append(sample)
	
		return np.array(batch_input)

	def __load_filenames(self):
		files = os.listdir(self.data_folder)
		files = list(filter(lambda x: x.endswith(".mat"), files))
		assert(len(files) > 0), "No .mat files found in data folder: {}".format(self.data_folder)

		print("Found {} .mat files in {}".format(len(files), self.data_folder))

		files = list(sorted(files))

		self.filenames = files
		# Expand all files to full path
		self.files = list(map(lambda x: os.path.join(self.data_folder, x), files))

class CNN_DataLoader(tf.keras.utils.Sequence):
	def __init__(self, data_folder, batch_size, sample_length=1024, skip_ahead=16):
		self.data_folder = data_folder
		self.skip_ahead = skip_ahead
		self.batch_size = batch_size
		self.sample_length = sample_length

		self.__load_filenames()

		self.current_file_index = 0
		self.__load_next_file_into_memory()

		self.__create_batch_indices()
		self.__shuffle_batch_indices()

	def __create_batch_indices(self):
		self.batch_indices = list(range(0, self.current_timeseries.shape[0] - self.sample_length, self.skip_ahead))

	def __shuffle_batch_indices(self):
		random.shuffle(self.batch_indices)

	def on_epoch_end(self):
		# Load new file from datafolder into memory and train.
		# New file in consecutive order of file array.
		# If training has seen all files --> shuffle file array and start again
		self.current_file_index = (self.current_file_index + 1) % len(self.files)

		if self.current_file_index == 0: # Shuffle
			random.shuffle(self.files)

		self.__load_next_file_into_memory()

		self.__create_batch_indices()
		self.__shuffle_batch_indices()

	def __normalize_sample(self, sample):
		max = np.max(sample)
		min = np.min(sample)
		return (sample - min) / (max - min)

	def __load_next_file_into_memory(self):
		file_to_load = self.files[self.current_file_index]
		print("Processing: {}".format(file_to_load))

		self.current_timeseries = loadmat(file_to_load)["Data"][..., 0, np.newaxis] # Take zero'th timeseries

	def __len__(self):
		return math.floor(len(self.batch_indices) // self.batch_size)

	def __getitem__(self, idx):
		batch_input = []
		batch_output = []
		for i in range(idx, idx + self.batch_size):
			current_index = self.batch_indices[i]
			sample = self.current_timeseries[current_index:current_index + self.sample_length]
			sample = self.__normalize_sample(sample)

			batch_input.append(sample)
			batch_output.append(sample)
	
		return np.array(batch_input), np.array(batch_output)

	def __load_filenames(self):
		files = os.listdir(self.data_folder)
		files = list(filter(lambda x: x.endswith(".mat"), files))
		assert(len(files) > 0), "No .mat files found in data folder: {}".format(self.data_folder)

		print("Found {} .mat files in {}".format(len(files), self.data_folder))

		# Expand all files to full path
		self.files = list(map(lambda x: os.path.join(self.data_folder, x), files))

class Enc_Dec_DataLoader(tf.keras.utils.Sequence):
	def __init__(self, data_folder, batch_size, sample_length=1024, skip_ahead=16):
		self.data_folder = data_folder
		self.skip_ahead = skip_ahead
		self.batch_size = batch_size
		self.sample_length = sample_length

		self.__load_filenames()

		self.current_file_index = 0
		self.__load_next_file_into_memory()

		self.__create_batch_indices()
		self.__shuffle_batch_indices()

	def __create_batch_indices(self):
		self.batch_indices = list(range(0, self.current_timeseries.shape[0] - self.sample_length, self.skip_ahead))

	def __shuffle_batch_indices(self):
		random.shuffle(self.batch_indices)

	def on_epoch_end(self):
		# Load new file from datafolder into memory and train.
		# New file in consecutive order of file array.
		# If training has seen all files --> shuffle file array and start again
		self.current_file_index = (self.current_file_index + 1) % len(self.files)

		if self.current_file_index == 0: # Shuffle
			random.shuffle(self.files)

		self.__load_next_file_into_memory()

		self.__create_batch_indices()
		self.__shuffle_batch_indices()

	def __normalize_sample(self, sample):
		max = np.max(sample)
		min = np.min(sample)
		return (sample - min) / (max - min)

	def __load_next_file_into_memory(self):
		file_to_load = self.files[self.current_file_index]
		print("Processing: {}".format(file_to_load))

		self.current_timeseries = loadmat(file_to_load)["Data"][..., 0, np.newaxis] # Take zero'th timeseries

	def __len__(self):
		return math.floor(len(self.batch_indices) // self.batch_size)

	def __getitem__(self, idx):
		batch_input = {"encoder_inputs": [], "decoder_inputs": []}
		batch_output = []
		for i in range(idx, idx + self.batch_size):
			current_index = self.batch_indices[i]
			sample = self.current_timeseries[current_index:current_index + self.sample_length]
			sample = self.__normalize_sample(sample)
			reversed_sample = np.copy(sample)[::-1]

			teacher_sample = np.copy(sample)[::-1]
			teacher_sample[1:] = teacher_sample[:-1]
			teacher_sample[0] = [0]

			batch_input["encoder_inputs"].append(sample)
			batch_input["decoder_inputs"].append(teacher_sample)

			batch_output.append(reversed_sample)
		batch_input["encoder_inputs"] = np.array(batch_input["encoder_inputs"])
		batch_input["decoder_inputs"] = np.array(batch_input["decoder_inputs"])

		return batch_input, np.array(batch_output)

	def __load_filenames(self):
		files = os.listdir(self.data_folder)
		files = list(filter(lambda x: x.endswith(".mat"), files))
		assert(len(files) > 0), "No .mat files found in data folder: {}".format(self.data_folder)

		print("Found {} .mat files in {}".format(len(files), self.data_folder))

		files = sorted(files)
		# Expand all files to full path
		self.files = list(map(lambda x: os.path.join(self.data_folder, x), files))
		
