import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os
from models.keras.cnn import build_cnn_model

def normalize_sample(sample):
	max = np.max(sample)
	min = np.min(sample)
	return (sample - min) / (max - min)

def main():
	window_size = 128
	filter_size = 3
	num_conv = 1
	model_weights = "trained_models/Lager4/ws128/skip_first_0/lat_2/cnn_model.h5"

	filter_num = 64 // 2

	model_parameters = {"window_size": window_size,
						"filter_num": filter_num,
						"filter_size": filter_size,
						"num_conv": num_conv}

	print("Building model")
	model = build_cnn_model(model_parameters)

	model.load_weights(model_weights)

	data_path = "Data/Lager4/complete_set/"
	mat_files = ["2019_02_28__11_52_15.mat", "2019_02_28__12_52_16.mat", "2019_02_28__22_01_15.mat", "2019_02_28__22_43_16.mat", "2019_03_01__02_13_16.mat", "2019_03_01__14_26_57.mat"]

	save_path = "reconstruction_plots"

	for file in mat_files:
		full_path = os.path.join(data_path, file)

		data = loadmat(full_path)["Data"][:, 0]

		os.makedirs(os.path.join(save_path, file), exist_ok=True)

		for i in range(10):
			start_index = np.random.randint(0, data.size - 128, 1)[0]
			sample = data[start_index:start_index+128]
			sample = normalize_sample(sample)
			original = np.copy(sample)
			sample = sample[None, :, None]

			predicted_sample = np.squeeze(model.predict(sample)[0][0], -1)

			mse = np.mean(np.square(original - predicted_sample))

			plt.title("MSE: {}".format(mse))
			plt.plot(original, label="Original Signal")
			plt.plot(predicted_sample, label="Reconstructed Signal")
			
			plt.legend()
			plt.savefig(os.path.join(save_path, file, "{}.png".format(i)))
			plt.clf()

if __name__ == "__main__":
	main()