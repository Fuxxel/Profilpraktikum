import numpy as np
# import matplotlib.pyplot as plt
import argparse
import os

def main(args):
	npy_files = os.listdir(args.input)
	npy_files_input = sorted(filter(lambda x: x.endswith("input.npy"), npy_files))
	npy_files_predicted = sorted(filter(lambda x: x.endswith("predicted.npy"), npy_files))

	metrics = []

	with open(args.output, "w") as results_csv:
		results_csv.write("Number,Abs. Error,Sq. Error,Global\n")
		for input_file, prediction_file in zip(npy_files_input, npy_files_predicted):
			input = np.load(os.path.join(args.input, input_file))
			prediction = np.load(os.path.join(args.input, prediction_file))

			abs_error = np.abs(input - prediction)
			squared_error = (input - prediction)**2

			global_error = np.sum(squared_error)

			abs_perc = np.percentile(abs_error, 99.5)
			squared_perc = np.percentile(squared_error, 99.5)

			results_csv.write("{},{},{},{}\n".format(input_file.split(".")[0], abs_perc, squared_perc, global_error))

			metrics.append([abs_perc, squared_perc, global_error])
			# ax_input_prediction = plt.subplot(311)
			# ax_input_prediction.plot(input, label="Observation")
			# ax_input_prediction.plot(prediction, label="Emission")
			# ax_input_prediction.set_xticklabels([])

			# ax_abs_error = plt.subplot(312)
			# ax_abs_error.plot(abs_error, color="green")
			# ax_abs_error.set_ylabel("Abs. Error")
			# ax_abs_error.set_xticklabels([])

			# ax_squared_error = plt.subplot(313)
			# ax_squared_error.plot(squared_error, color="red")
			# ax_squared_error.set_ylabel("Sq. Error")
			# ax_squared_error.set_xlabel("Index")

			# plt.show()
	
		means = np.mean(np.array(metrics), axis=0)
		results_csv.write(",{},{},{}\n".format(means[0], means[1], np.percentile(np.array(metrics)[:, 2], 99.5)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input", type=str, required=True, help="Path to .npy data files from tests.")
	parser.add_argument("-o", "--output", type=str, required=True, help="Path for output files.")
	
	args = parser.parse_args()

	main(args)