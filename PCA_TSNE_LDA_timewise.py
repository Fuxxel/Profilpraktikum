import numpy as np
from scipy.io import loadmat
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

input_path = "Data/Lager4/complete_set"

files = os.listdir(input_path)
files = filter(lambda x: x.endswith(".mat"), files)
files = list(sorted(files))
files = [os.path.join(input_path, x) for x in files]

dataset = []

num_files = len(files)
for i, file in enumerate(files):
	print("\r{}/{}".format(i + 1, num_files), end="")
	data = loadmat(file)["Data"][:, 0]
	dataset.append(data)
print("")
dataset = np.array(dataset)

print("Applying PCA...")
transformed_PCA = PCA(n_components=2).fit_transform(dataset)
np.save("pca_transformed_lager4.npy", transformed_PCA)

print("Applying TSNE...")
transformed_TSNE = TSNE(n_components=2).fit_transform(dataset)
np.save("tsne_transformed_lager4.npy", transformed_TSNE)

# print("Applying LDA...")
# transformed_LDA = LDA(n_components=2).fit_transform(dataset)
# np.save("lda_transformed_lager4.npy", transformed_LDA)