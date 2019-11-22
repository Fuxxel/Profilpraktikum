import numpy as np
import os 
import matplotlib.pyplot as plt

window = 1024

def main():
	path = "model_results/Lager4/enc_dec/ws_100/lat_256/"

	folders = os.listdir(path)
	folders = filter(lambda x: x.endswith(".mat"), folders)
	folders = sorted(folders)

	metrics = []

	for folder in folders:
		folder_path = os.path.join(path, folder)
		gt_path = os.path.join(folder_path, "complete_timeseries_input.npy")
		pred_path = os.path.join(folder_path, "complete_timeseries_predicted.npy")

		gt = np.load(gt_path) #[:2048]
		pred = np.load(pred_path) #[:2048]

		# plt.plot(gt)
		# plt.plot(pred)
		# plt.show()

		abs_error = np.quantile(np.abs(gt - pred), 0.995)
		sq_error = np.quantile((gt - pred)**2, 0.995)
		global_error = np.sum((gt - pred)**2)

		metrics.append([abs_error, sq_error, global_error])

		print("{}: abs:{:.4f}, sq:{:.4f}, global:{:.4f}".format(folder, abs_error, sq_error, global_error))

	metrics = np.array(metrics)
	plt.plot(metrics[:,0], label="Abs error")
	plt.plot(metrics[:,1], label="Sq error")
	plt.legend()
	plt.title("Abs. and sq. error")
	# plt.plot(metrics[:,2], label="Global error")
	axes = plt.gca()
	axes.set_xticklabels([":".join(x.split("_")[-3:])[:-4] for x in folders])
	plt.show()


if __name__ == "__main__":
	main()