from tslearn.metrics import dtw
import numpy as np
from scipy.io import loadmat
import os 
import ntpath
from multiprocessing import Pool

window = 6144
threads = 20

save = "dtw/Lager4/old_normalized"
input_path = "Data/Lager4/complete_set"

def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

def normalize(data):
	ma = np.max(data)
	mi = np.min(data)
	return (data - mi) / (ma - mi)

def process_file(args):
	file, reference_data = args
	global save

	data = normalize(loadmat(file)["Data"][:, 0])

	error = 0.0
	steps = data.shape[0] // window
	for i in range(0, data.shape[0], window):
		# print("\r{}/{}".format((i + 1) // window, steps), end="")
		compare_to = data[i:i+window]
		error += dtw(reference_data, compare_to)
	# print("")
	# print(error)
	filenname = path_leaf(file)
	with open(os.path.join(save, filenname.split(".")[0] + ".dtw"), "w") as dtw_out:
		dtw_out.write("{}".format(error))

def main():
	global threads, input_path, save

	os.makedirs(save, exist_ok=True)

	files = os.listdir(input_path)
	files = filter(lambda x: x.endswith(".mat"), files)
	files = list(sorted(files))
	files = [os.path.join(input_path, x) for x in files]

	# load reference data:
	reference_data = normalize(loadmat(files[0])["Data"][:, 0])[window*100:window*101]
	reference_data_copies = [np.copy(reference_data) for _ in range(len(files))]

	args = zip(files, reference_data_copies)

	p = Pool(threads)
	p.map(process_file, args)


if __name__ == "__main__":
	main() 