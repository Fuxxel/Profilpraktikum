from scipy.io import loadmat
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.utilities.dataframe_functions import impute

import numpy as np
import pandas 
from itertools import product

import argparse
import os
from multiprocessing import Pool
import ntpath

def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

def process_file(file):
	data = loadmat(file)["Data"][:,0]

	data_dict = {"time": np.arange(0, len(data)), "x": data}

	df = pandas.DataFrame.from_dict(data_dict)
	df["id"] = "A"

	params = {
		# "abs_energy": None,
        # "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
		# "agg_linear_trend": [{"attr": attr, "chunk_len": i, "f_agg": f}
        #                          for attr in ["rvalue", "intercept", "slope", "stderr"]
        #                          for i in [5, 10, 50]
        #                          for f in ["max", "min", "mean", "var"]],
		# "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(5) for k in [10]],
		"autocorrelation": [{"lag": x} for x in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]],
		# "binned_entropy": [{"max_bins": max_bins} for max_bins in [10]],
		# "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
		# "fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(100))],
		# "kurtosis": None,
		# "mean": None,
		# "mean_abs_change": None,
		# "mean_change": None,
		# "median": None,
		# "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
		# "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
		# "skewness": None,
		# "standard_deviation": None,
		# "variance": None
	}

	features = extract_features(df, 
								column_id="id", 
								column_sort="time", 
								column_kind=None, 
								default_fc_parameters=params,
								n_jobs=0,
								disable_progressbar=True)
	
	file = path_leaf(file)
	filename = "".join(file.split(".")[:-1]) + ".csv"
	full_save_path = os.path.join(args.output_path, filename)
	features.to_csv(full_save_path)

def main(args):
	os.makedirs(args.output_path, exist_ok=True)

	files = os.listdir(args.input_path)
	files = filter(lambda x: x.endswith(".mat"), files)
	files = sorted(files)
	files = list(map(lambda x: os.path.join(args.input_path, x), files))

	for file in files:
		process_file(file)

	# Multiprocessing:
	# process_file(files[args.n_file])

	# print(files)

	# num_files = len(files)

	# p = Pool(args.n_jobs)

	# p.map(process_file, files)

	# for file in files:
	# 	full_file_path = os.path.join(args.input_path, file)
	# 	features = process_file(full_file_path, args.n_jobs, args.chunksize)

	# 	filename = "".join(file.split(".")[:-1]) + ".features"
	# 	full_save_path = os.path.join(args.output_path, filename)
	# 	features.to_pickle(full_save_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# parser.add_argument("-n", "--n_file", type=int, default=1, help="File to work on in file list.")
	
	parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to .mat training files.")
	parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save directory.")
	
	args = parser.parse_args()

	main(args)