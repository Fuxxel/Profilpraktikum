import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
	npy_files = os.listdir(args.input_path)
	npy_files_input = sorted(filter(lambda x: x.endswith("input.npy"), npy_files))
	npy_files_predicted = sorted(filter(lambda x: x.endswith("predicted.npy"), npy_files))

	for input_file, prediction_file in zip(npy_files_input, npy_files_predicted):
		input = np.load(input_file)
		prediction = np.load(prediction_file)

		abs_error = np.abs(input - prediction)
		squared_error = (input - prediction)**2

		ax_input_prediction = plt.subplot(311)
		ax_input_prediction.plot(input, label="Observation")
		ax_input_prediction.plot(prediction, label="Emission")
		ax_input_prediction.set_xticklabels([])

		ax_abs_error = plt.subplot(312)
		ax_abs_error.plot(abs_error, color="green")
		ax_abs_error.xlabel("Abs. Error")
		ax_abs_error.set_xticklabels([])

		ax_squared_error = plt.subplot(312)
		ax_squared_error.plot(squared_error, color="red")
		ax_squared_error.xlabel("Sq. Error")
		ax_squared_error.xlabel("Index")

		plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to .npy data files from tests.")
	parser.add_argument("-o", "--output", type=str, required=True, help="Path for output files.")
	
	args = parser.parse_args()

	main(args)