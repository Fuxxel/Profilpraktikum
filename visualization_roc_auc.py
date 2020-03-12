import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(16, 9), dpi=80 )

lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')

plt.tight_layout()
plt.show()