import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow import keras

import numpy as np
import ntpath
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

def load_and_parse_txt_list(path):
    result = []
    with open(path) as f:
        for line in f:
            c, p = line.split(":")
            result.append((int(c.lstrip().rstrip()), p.lstrip().rstrip()))

    return result

def normalize_window(window):
    min = window.min()
    return (window - min) / (window.max() - min)

def cut_into_windows(sample, window_size):
    windowed_data = []
    current_window_index = 0
    increment = window_size // 2
    while current_window_index < sample.shape[0] - window_size:
        window = normalize_window(sample[current_window_index:current_window_index + window_size])
        windowed_data.append(window[..., np.newaxis])
        current_window_index += increment

    return windowed_data

def load_ims_data(files_and_classes, window_size, offset, skip):
    complete_data_x = []
    complete_data_y = []
    
    files_and_classes = sorted(files_and_classes, key=lambda x: x[1])[offset::skip]
    for c, file_path in files_and_classes:
        file_data = np.load(file_path)
        if "Run1" in file_path:
            file_data = file_data[0]

        windowed_data = cut_into_windows(file_data, window_size)
        complete_data_x.extend(windowed_data)
        complete_data_y.extend([c for _ in range(len(windowed_data))])

    complete_data_x = np.array(complete_data_x)
    complete_data_y = np.array(complete_data_y)

    assert complete_data_x.shape[0] == complete_data_y.shape[0], "Data loader failed"

    return complete_data_x, complete_data_y 

def load_ims_data_training(files_and_classes, window_size):
    return load_ims_data(files_and_classes, window_size, offset=0, skip=2)

def load_ims_data_test(files_and_classes, window_size):
    return load_ims_data(files_and_classes, window_size, offset=1, skip=2)

def load_ims_data_x_single_file(file, window_size):    
    file_data = np.load(file)
    if "Run1" in file:
        file_data = file_data[0]

    windowed_data = cut_into_windows(file_data, window_size)
    return np.array(windowed_data)

def build_model(window_size):
    model = keras.Sequential([
        Input((window_size, 1)),
        Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        MaxPool1D(pool_size=2),

        Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
        MaxPool1D(pool_size=2),

        Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"),
        MaxPool1D(pool_size=2),

        Flatten(),

        Dense(100, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

def classification_from_preditions(predictions):
    return np.mean(predictions)

def timestamp_to_datestring(timestamp):
    date = ".".join(timestamp.split(".")[:3][::-1])
    return date + " " + ":".join(timestamp.split(".")[-4:-1])

def main():
    batch_size = 1024*2
    window_size = 1024

    base_path = "IMSDataset_extracted/Run3/Lager4"

    classes_and_filenames = load_and_parse_txt_list("IMSDataset_classification/Run3/Lager4/classes.txt")
    classes_and_filenames = [(c, os.path.join(base_path, filename)) for c, filename in classes_and_filenames]

    model = keras.models.load_model("classify_ims_run_2_bearing_1_model.h5")
    print(model.summary())

    plot_data = []
    for i, (real_class_value, test_file) in enumerate(classes_and_filenames):
        print(f"\r{i + 1}/{len(classes_and_filenames)}", end="")
        x_data = load_ims_data_x_single_file(test_file, window_size=window_size)
        predictions = model.predict(x_data, batch_size=batch_size)
        predicted_class_value = classification_from_preditions(predictions)
        
        plot_data.append((timestamp_to_datestring(path_leaf(test_file)), predicted_class_value, real_class_value))
    print()

    np.save("classify_ims_transfer_from_run_2_bearing_1_to_run_3_bearing_4.npy", np.array(plot_data))

if __name__ == "__main__":
    main()