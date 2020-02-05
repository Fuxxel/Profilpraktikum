import os
import numpy as np
import matplotlib.pyplot as plt

dtw_path = "dtw/average/Lager4"

files = os.listdir(dtw_path)
files = filter(lambda x: x.endswith(".dtw"), files)

values = []
for file in files:
	full_path = os.path.join(dtw_path, file)

	values.append(np.loadtxt(full_path))

plt.plot(np.array(values))
plt.savefig("dtw_average_lager4.png")