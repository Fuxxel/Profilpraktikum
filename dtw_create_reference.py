import os
from scipy.io import loadmat
import numpy as np

def normalize(data):
	ma = np.max(data)
	mi = np.min(data)
	return (data - mi) / (ma - mi)

input_path = "Data/Lager5/complete_set"
save_path = "dtw/reference"
window = 6144

os.makedirs(save_path, exist_ok=True)

files = os.listdir(input_path)
files = filter(lambda x: x.endswith(".mat"), files)
files = list(sorted(files))
files = [os.path.join(input_path, x) for x in files]

# load reference data:
print("Reference Files:")
reference_data = []
for i in range(5):
	print(files[i])
	data = loadmat(files[i])["Data"][:, 0]
	
	max_index = data.shape[0] - window
	min_index = window * 10

	sample_points = np.random.randint(min_index, max_index, 10)

	for point in sample_points:
		sample_data = normalize(data[point:point + window])
		reference_data.append(sample_data)

reference_data = np.array(reference_data)
np.save(os.path.join(save_path, "lager5.npy"), reference_data)