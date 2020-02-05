import numpy as np
from scipy.io import loadmat
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

switch = 32
downsample_folders = 10
downsample = 5

input_path = "latent_vectors/Lager4/ws128/filt_{}".format(switch)

folders = os.listdir(input_path)
folders = filter(lambda x: x.endswith(".mat"), folders)
folders = list(sorted(folders))
folders = [os.path.join(input_path, x) for x in folders]
folders = folders[::downsample_folders]

dataset = []

for i, folder in enumerate(folders):
	print("\r{}/{}".format(i + 1, len(folders)), end="")
	file_path = os.path.join(folder, "latent_spaces.npy")
	data = np.load(file_path)[::downsample]
	data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
	data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
	dataset.append(data)

print("")
dataset = np.array(dataset)
dataset = np.reshape(dataset, (dataset.shape[0] * dataset.shape[1], dataset.shape[2]))

print("Applying PCA...")
transformed_PCA = PCA(n_components=2).fit_transform(dataset)
np.save("pca_transformed_latent_filt_{}_lager4_downsample_folder_10_downsample_5.npy".format(switch), transformed_PCA)

print("Applying TSNE...")
transformed_TSNE = TSNE(n_components=2).fit_transform(dataset)
np.save("tsne_transformed_latent_filt_{}_lager4_downsample_folder_10_downsample_5.npy".format(switch), transformed_TSNE)

# print("Applying LDA...")
# transformed_LDA = LDA(n_components=2).fit_transform(dataset)
# np.save("lda_transformed_lager4.npy", transformed_LDA)