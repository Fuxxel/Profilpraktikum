import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os
from models.keras.cnn import build_cnn_model
from itertools import product

def normalize_sample(sample):
	max = np.max(sample)
	min = np.min(sample)
	return (sample - min) / (max - min)

def main():
	filter_size = 3
	num_conv = 1
	
	latents = [2, 4, 8]
	window_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

	for latent, window_size in product(latents, window_sizes):
		print(latent, " ", window_size)
		model_weights = "trained_models/Lager4/ws{}/skip_first_0/lat_{}/cnn_model.h5".format(window_size, latent)

		filter_num = 64 // latent

		model_parameters = {"window_size": window_size,
							"filter_num": filter_num,
							"filter_size": filter_size,
							"num_conv": num_conv}

		print("Building model")
		model = build_cnn_model(model_parameters)

		model.load_weights(model_weights)

		# Lager 4
		# data_path = "Data/Lager4/complete_set/"
		# mat_files = ["2019_02_28__11_52_15.mat", "2019_02_28__12_52_16.mat", "2019_02_28__22_01_15.mat", "2019_02_28__22_43_16.mat", "2019_03_01__02_13_16.mat", "2019_03_01__14_26_57.mat"]
		# save_path = "reconstruction_plots/Lager4/ws{}/lat_{}/".format(window_size, latent)

		################
		# Lager 5
		data_path = "Data/Lager5/complete_set/"
		mat_files = ["2019_03_13__14_17_17.mat",
					 "2019_03_13__19_19_16.mat",
					 "2019_03_14__02_01_17.mat",
					 "2019_03_14__18_40_12.mat",
					 "2019_03_15__04_40_12.mat",
					 "2019_03_15__10_10_12.mat",
					 "2019_03_15__16_50_11.mat",
					 "2019_03_16__01_20_11.mat",
					 "2019_03_16__04_50_11.mat",
					 "2019_03_16__12_04_13.mat",
					 "2019_03_16__17_40_13.mat",
					 "2019_03_16__20_28_13.mat",
					 "2019_03_17__00_42_13.mat"]
		save_path = "reconstruction_plots/Lager5/ws{}/lat_{}/".format(window_size, latent)
		################

		for file in mat_files:
			print(file)
			full_path = os.path.join(data_path, file)

			data = loadmat(full_path)["Data"][:, 0]

			os.makedirs(os.path.join(save_path, file), exist_ok=True)

			for i in range(10):
				start_index = np.random.randint(0, data.size - window_size, 1)[0]
				sample = data[start_index:start_index + window_size]
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