import numpy as np
import os 
import matplotlib.pyplot as plt
import pickle
import argparse 

window = 1024

def main(args):
	path = "model_results/Lager5/whole_dataset/cnn/predicted_with_correct_net/skip_29/filt_{}/".format(args.switch)
	raw_save = "lager5_global_error_cnn_whole_dataset_skip_29_filt_{}".format(args.switch)
	# save = "lager4_global_error_cnn_whole_dataset_filt_32.png"

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
	np.save(raw_save + ".npy", metrics)

	dates = [".".join(x.split("_")[1:3][::-1]) for x in folders]
	ticks = [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(folders, dates)]
	with open(raw_save + ".ticks", "wb") as dump_file:
		pickle.dump(ticks, dump_file)

	# np.save(raw_save, metrics)
	# plt.figure(figsize=(11.27*7, 7.04), dpi=227)
	# # plt.plot(metrics[:,0], label="Abs error")
	# # plt.plot(metrics[:,1], label="Sq error")
	# # plt.legend()
	# # plt.title("Abs. and sq. error")
	# plt.title("Global error")
	# # plt.locator_params(axis="x", nbins=len(folders))
	# plt.plot(metrics[:,2], label="Global error")
	# dates = [".".join(x.split("_")[1:3][::-1]) for x in folders]
	# plt.xticks(np.arange(len(folders)), [y + ". " + ":".join(x.split("_")[-3:])[:-4] for x,y in zip(folders, dates)])
	# ax = plt.gca()
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	# 		 rotation_mode="anchor")
	# # axes = plt.gca()
	# # axes.set_xticklabels([":".join(x.split("_")[-3:])[:-4] for x in folders])
	# # plt.show()
	# plt.tight_layout()
	# plt.savefig(save)
	# plt.clf()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-s", "--switch", type=str)
	
	args = parser.parse_args()

	main(args)