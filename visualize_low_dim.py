import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import cm

data = np.load("pca_transformed_lager4.npy")

# cm = matplotlib.cm.get_cmap()
# colors = cm(np.linspace(0, 1, data.shape[0]))

cax = plt.scatter(data[:, 0], data[:, 1], c=range(data.shape[0]), cmap=cm.get_cmap('RdYlBu'))
# vmin=0, vmax=data.shape[0], s=35
plt.colorbar(cax)

plt.show()