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
	j = 0
	for ref in references:
		error += dtw(np.array(ref), normalize(window_data))
		j += 1
	
	return error / j

def process_file(file, reference_data):
	global save
	print("file: {}".format(file))

	data = loadmat(file)["Data"][:, 0]
	print("data length: {}".format(data.shape[0]))
	data = data[:(data.shape[0] // window) * window]
	print("data length aligned: {}".format(data.shape[0]))

	window_chunks = chunks(data, window)
	num_chunks = data.shape[0] // window
	print("num chunks: {}".format(num_chunks))
	reference_copies = [reference_data for _ in range(num_chunks)]
	print("reference copies: {}".format(np.array(reference_copies).shape))

	args = zip(window_chunks, reference_copies)

	p = Pool(threads)
	errors = p.map(process_window, args)
	print("num errors: {}".format(len(errors)))
	print("errors: {}".format(errors))
	error = sum(errors)
	
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

	print("Starting...")
	process_file(files[args.file_num], reference_data)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--file_num", type=int)
	
	args = parser.parse_args()

	main(args)