import os
from itertools import product
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

def load_txt_list(path):
	result = []
	with open(path, "r") as txt_file:
		for line in txt_file:
			result.append(line.rstrip().lstrip())
	return result

def main():
	lagers = ["Lager4", "Lager5"]
	window_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
	latent_sizes = [2, 4, 8]

	for lager, window_size, latent_size in product(lagers, window_sizes, latent_sizes):
		collected_latents = []
		print(lager, window_size, latent_size)
		save_folder = os.path.join("latent_plots", lager, "ws{}".format(window_size), "nc1", "fs3", "lat_{}".format(latent_size))
		os.makedirs(save_folder, exist_ok=True)

		path_to_folders = os.path.join("model_predictions", lager, "ws{}".format(window_size), "nc1", "fs3", "skip_first_0", "lat_{}".format(latent_size))
		folders = os.listdir(path_to_folders)
		folders = filter(lambda x: os.path.isdir(os.path.join(path_to_folders, x)), folders)

		roi_list = load_txt_list(os.path.join("roi", lager + ".txt"))

		roi_folders = list(filter(lambda x: x in roi_list, folders))
		roi_folders = sorted(roi_folders)

		for i in range(0, len(roi_folders), 5):
			folder = roi_folders[i]
			latents = np.load(os.path.join(path_to_folders, folder, "latent_spaces.npy"))

			rand_iterations = (np.random.rand(10) * latents.shape[0]).astype(np.int)
			for it in rand_iterations:
				rand_batches = (np.random.rand(10) * latents.shape[1]).astype(np.int)
				for batch in rand_batches:
					collected_latents.append(latents[it, batch])

		print("Collected {} latent space vectors".format(len(collected_latents)))
		collected_latents = np.array(collected_latents)
		collected_latents = np.reshape(collected_latents, (collected_latents.shape[0], np.prod(collected_latents.shape[1:])))

		transformed = PCA(n_components=2).fit_transform(collected_latents)

		cax = plt.scatter(transformed[:, 0], transformed[:, 1], c=range(transformed.shape[0]), cmap=cm.get_cmap('RdYlBu'), alpha=0.2)
		plt.colorbar(cax)

		plt.savefig(os.path.join(save_folder, "pca.png"))
		plt.clf()

		for perplexity in [30, 100, 200, 400, 600, 800]:
			transformed = TSNE(n_components=2, perplexity=perplexity).fit_transform(collected_latents)

			cax = plt.scatter(transformed[:, 0], transformed[:, 1], c=range(transformed.shape[0]), cmap=cm.get_cmap('RdYlBu'), alpha=0.2)
			plt.colorbar(cax)

			plt.savefig(os.path.join(save_folder, "tsne_{}.png".format(perplexity)))
			plt.clf()

if __name__ == "__main__":
	main()