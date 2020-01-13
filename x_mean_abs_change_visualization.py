import numpy as np
import matplotlib.pyplot as plt

def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))

# modulator_frequency = 10.0
# carrier_frequency = 40.0
# modulation_index = 1.0

time = np.arange(44100.0) / 44100.0
# modulator_am = time**2
# carrier = np.sin(2.0 * np.pi * carrier_frequency * time)

# product_am = modulator_am * carrier

# modulator_fm = np.sin(2.0 * np.pi * modulator_frequency * time) * modulation_index
# product_fm = np.zeros_like(modulator_fm)
# for i, t in enumerate(time):
#     product_fm[i] = np.sin(2. * np.pi * (carrier_frequency * t + modulator_fm[i]))

# plt.subplot(2, 2, 1)
# plt.title('Amplitude Modulation')
# plt.plot(modulator_am)
# plt.subplot(2, 2, 3)
# plt.plot(product_am, label="MAC={:.3E}".format(mean_abs_change(product_am)))
# plt.ylabel('Amplitude')
# plt.xlabel('Output signal')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.title('Frequency Modulation')
# plt.plot(modulator_fm)
# plt.subplot(2, 2, 4)
# plt.plot(product_fm, label="MAC={:.3E}".format(mean_abs_change(product_fm)))
# plt.ylabel('Amplitude')
# plt.xlabel('Output signal')
# plt.legend()
# plt.show()

macs = []
for f in range(1, 200):
    x = np.sin(np.pi * time * f)
    print(mean_abs_change(x))
    macs.append(mean_abs_change(x))

plt.subplot(1, 2, 1)
plt.title('Frequency')
plt.plot(macs)

macs.clear()
for a in range(1, 200):
    x = np.sin(np.pi * time) * a
    print(mean_abs_change(x))
    macs.append(mean_abs_change(x))

plt.subplot(1, 2, 2)
plt.title('Amplitude')
plt.plot(macs)
plt.show()