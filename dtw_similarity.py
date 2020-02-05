from tslearn.metrics import dtw
import numpy as np
from scipy.io import loadmat
import os 
import ntpath
from multiprocessing import Pool
import argparse
import math

window = 6144
threads = 20

reference_path = "dtw/reference/lager4.npy"
save = "dtw/average/Lager4/"
input_path = "Data/Lager4/complete_set"

def chunks(lst, n):
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def normalize(data):
	ma = np.max(data)
	mi = np.min(data)
	return (data - mi) / (ma - mi)

def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

def process_window(args):
	window_data, references = args

	error = 0
	for j, ref in enumerate(references):
		# print("\t{}".format(j))
		# print(np.array(ref).shape)
		# print(compare_to.shape)
		error += dtw(np.array(ref), window_data)
	
	return error

def process_file(args):
	file, reference_data = args
	global save

	data = loadmat(file)["Data"][:, 0]
	data = data[:(data.shape[0] // window) * window]

	window_chunks = chunks(data, window)
	num_chunks = math.ceil(data.shape[0] / window)
	reference_copies = [reference_data for _ in range(num_chunks)]

	args = zip(window_chunks, reference_copies)

	p = Pool(threads)
	errors = p.map(process_window, args)
	print(len(errors))
	errors = sum(errors)

	error = errors / (num_chunks * (5 * 10))
	
	# for i in range(0, data.shape[0], window):
	# 	print(i)
	# 	compare_to = normalize(data[i:i+window])

	# 	for j, ref in enumerate(reference_data):
	# 		print("\t{}".format(j))
	# 		print(np.array(ref).shape)
	# 		print(compare_to.shape)
	# 		errors.append(dtw(np.array(ref), compare_to))

	# error = sum(errors) / len(errors)
	
	filenname = path_leaf(file)
	with open(os.path.join(save, filenname.split(".")[0] + ".dtw"), "w") as dtw_out:
		dtw_out.write("{}".format(error))

def main(args):
	global threads, input_path, save

	os.makedirs(save, exist_ok=True)

	files = os.listdir(input_path)
	files = filter(lambda x: x.endswith(".mat"), files)
	files = list(sorted(files))
	files = [os.path.join(input_path, x) for x in files]

	# load reference data:
	reference_data = np.load(reference_path)
	# print("Reference Files:")
	# reference_data = []
	# for i in range(5):
	# 	print(files[i])
	# 	data = loadmat(files[i])["Data"][:, 0]
		
	# 	max_index = data.shape[0] - window
	# 	min_index = window * 10

	# 	sample_points = np.random.randint(min_index, max_index, 10)

	# 	for point in sample_points:
	# 		sample_data = normalize(data[point:point + window])
	# 		reference_data.append(sample_data)

	# reference_data = np.array(reference_data)

	print("Starting...")
	process_file((files[args.file_num], reference_data))

	# args = zip(files, reference_data_copies)

	# p = Pool(threads)
	# p.map(process_file, args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--file_num", type=int)
	
	args = parser.parse_args()

	main(args)