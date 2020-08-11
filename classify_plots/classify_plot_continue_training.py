import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

stats = pickle.load(open("classify_continue_training_stats.p", "rb"))

loss_bearing_5 = stats["loss_bearing_5"]
acc_bearing_5 = stats["acc_bearing_5"]
acc_bearing_4 = stats["acc_bearing_4"]

matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(16,9), dpi=200)

color = "tab:red"
ax1 = fig.gca()

ax1.plot(acc_bearing_4, label="Acc Bearing 4", color=color)
ax1.plot(acc_bearing_5, label="Acc Bearing 5", linestyle="dashed", color=color)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc="center right")

color = "tab:blue"
ax2 = ax1.twinx()
ax2.plot(loss_bearing_5, label="Loss Bearing 5", color=color)
ax2.set_ylabel("BCE Loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Continue training on Bearing 5")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("classify_plot_continue_training_on_bearing_5_stats.png")