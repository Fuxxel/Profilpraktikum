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
    
def find_majority_with_confidence(votes, confidences):
    scores = [0, 0]
    for vote, confidence in zip(votes, confidences):
        scores[vote] += confidence
    
    majority_vote = 0 if scores[0] > scores[1] else 1
    confidence = scores[majority_vote] / sum(scores)
    return majority_vote, confidence

def classification_from_preditions(predictions):
    return np.mean(predictions)

def timestamp_to_datestring(timestamp):
    date = ".".join(timestamp.split(".")[:3][::-1])
    return date + " " + ":".join(timestamp.split(".")[-4:-1])

def main():
    epochs = 50
    batch_size = 1024
    window_size = 1024

    base_path = "IMSDataset_extracted/Run2/Lager1"

    classes_and_filenames = load_and_parse_txt_list("IMSDataset_classification/Run2/Lager1/classes.txt")
    classes_and_filenames = [(c, os.path.join(base_path, filename)) for c, filename in classes_and_filenames]

    model = build_model(window_size=window_size)
    print(model.summary())

    train_data_x, train_data_y = load_ims_data_training(classes_and_filenames, window_size=window_size)
    test_data_x, test_data_y = load_ims_data_test(classes_and_filenames, window_size=window_size)

    train_data_x, val_data_x, train_data_y, val_data_y = train_test_split(train_data_x, train_data_y)

    dataset_train = tf.data.Dataset.from_tensor_slices((train_data_x, train_data_y)).shuffle(batch_size * 2).batch(batch_size)
    dataset_test = tf.data.Dataset.from_tensor_slices((test_data_x, test_data_y)).batch(batch_size)
    dataset_val = tf.data.Dataset.from_tensor_slices((val_data_x, val_data_y)).batch(batch_size)

    model.fit(dataset_train, epochs=epochs, validation_data=dataset_val)

    model.save("classify_ims_run_2_bearing_1_model.h5")
    
    print("Evaluate")
    metrics = model.evaluate(dataset_test)

    plot_data = []
    test_files = classes_and_filenames[1::2]
    for real_class_value, test_file in test_files:
        x_data = load_ims_data_x_single_file(test_file, window_size=window_size)
        predictions = model.predict(x_data)
        predicted_class_value = classification_from_preditions(predictions)
        
        plot_data.append((timestamp_to_datestring(path_leaf(test_file)), predicted_class_value, real_class_value))

    np.save("classify_ims_run_2_bearing_1.npy", np.array(plot_data))

if __name__ == "__main__":
    main()