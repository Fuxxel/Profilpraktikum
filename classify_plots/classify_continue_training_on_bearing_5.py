import torch
import torch.nn as nn
from torch.optim import AdamW

import numpy as np
from scipy.io import loadmat
import os
import glob
import random
import pickle
import matplotlib.pyplot as plt

def load_txt_list(path):
    result = []
    with open(path) as f:
        for line in f:
            result.append(line.lstrip().rstrip())

    return result

class DataLoader:
    def __init__(self, data_files, batch_size, sensor, sample_length, normalize_then_split=False):
        self.data_files = data_files
        self.batch_size = batch_size
        self.sample_length = sample_length
        self.sensor = sensor
        self.normalize_then_split = normalize_then_split
        self.category_balance = [0, 0]
        
        self.preloaded_data = []
        self.preloaded_gt = []
        self.__preload_data()
        self.preloaded_data = torch.from_numpy(self.preloaded_data)
        self.preloaded_gt = torch.from_numpy(self.preloaded_gt)

        self.__create_batch_indices()
        self.__shuffle_batch_indices()

    def __preload_data(self):
        print("Preloading data")
        for i, (data_file, category) in enumerate(self.data_files):
            print(f"\rLoading {i + 1}/{len(self.data_files)}", end="")
            ts = loadmat(data_file)["Data"][..., self.sensor, np.newaxis]

            if self.normalize_then_split:
                ts = self.__normalize_sample(ts)

            num_chunks = ts.shape[0] // self.sample_length
            chunks = np.array_split(ts, num_chunks)
              
            for chunk in chunks:
                if chunk.shape[0] != self.sample_length:
                    continue
                if not self.normalize_then_split:
                    chunk = self.__normalize_sample(chunk)
                self.preloaded_data.append(chunk)
                self.preloaded_gt.append(category)
                self.category_balance[category] += 1
        print()
        print(f"Category balance: 0={self.category_balance[0]}, 1={self.category_balance[1]}")

        self.preloaded_data = np.asarray(self.preloaded_data)
        self.preloaded_gt = np.asarray(self.preloaded_gt)

    def get_category_balance(self):
        return self.category_balance

    def __create_batch_indices(self):
        self.batch_indices = np.arange(len(self.preloaded_data))

    def __shuffle_batch_indices(self):
        random.shuffle(self.batch_indices)

    def __normalize_sample(self, sample):
        max = sample.max()
        min = sample.min()
        return (sample - min) / (max - min)

    def __len__(self):
        return len(self.batch_indices) // self.batch_size

    def __getitem__(self, idx):
        idx = idx * self.batch_size

        indices = torch.from_numpy(self.batch_indices[idx:idx + self.batch_size])
        batch_input = self.preloaded_data[indices].float()
        batch_output = self.preloaded_gt[indices].long()

        if batch_input.shape[0] != self.batch_size or batch_output.shape[0] != self.batch_size:
            raise IndexError()
    
        return batch_input.cuda(), batch_output.cuda()

class ClassifierNN(nn.Module):
    def __init__(self, input_length):
        super(ClassifierNN, self).__init__()

        self.input_length = input_length

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 16, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        self.dense = nn.Linear(16 * self.input_length // 2 // 2 // 2, 100)
        self.dense_out = nn.Linear(100, 2)

    def forward(self, input):
        input = self.relu(self.conv1(input))
        input = self.pool1(input)

        input = self.relu(self.conv2(input))
        input = self.pool2(input)
        
        input = self.relu(self.conv3(input))
        input = self.pool3(input)
        input = input.view(-1, self.num_flat_features(input))

        input = self.relu(self.dense(input))
        return self.sigmoid(self.dense_out(input))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def accuracy_metric(y_pred, y_true):
    maxed = torch.argmax(torch.softmax(y_pred, axis=-1), axis=-1)
    same = y_true == maxed
    return (same.sum().sum().item() / y_true.numel())

def normalize_sample(sample):
        max = sample.max()
        min = sample.min()
        return (sample - min) / (max - min)

def expand_and_annotate_list_bearing_5(txt_list, gt_list):
    result = []
    for entry in txt_list:
        result.append([os.path.abspath(os.path.join("Data/Lager5/complete_set/", entry)), 1 if entry in gt_list else 0])
    return result

def expand_and_annotate_list_bearing_4(txt_list, gt_list):
    result = []
    for entry in txt_list:
        result.append([os.path.abspath(os.path.join("Data/Lager4/complete_set/", entry)), 1 if entry in gt_list else 0])
    return result

def predict_test_batch(batch, network):
    chunks = np.asarray(batch)
    chunks = torch.from_numpy(chunks).float().cuda()
    prediction = network(chunks.permute(0, 2, 1))
    softmaxed = torch.softmax(prediction, axis=-1)
    confidence = softmaxed.max(-1).values.cpu()
    predicted_classes = torch.argmax(softmaxed, axis=-1).cpu()
    return predicted_classes.numpy().tolist(), confidence.numpy().tolist()
    
def find_majority_with_confidence(votes, confidences):
    scores = [0, 0]
    for vote, confidence in zip(votes, confidences):
        scores[vote] += confidence
    
    majority_vote = 0 if scores[0] > scores[1] else 1
    confidence = scores[majority_vote] / sum(scores)
    return majority_vote, confidence

def main():
    epochs = 50

    roi_list_bearing_5 = load_txt_list("roi/Lager5.txt")
    gt_list_bearing_5 = load_txt_list("gt/Lager5.txt")

    train_list_bearing_5 = roi_list_bearing_5[::2]
    test_list_bearing_5 = roi_list_bearing_5[1::2]
    val_list_bearing_5 = train_list_bearing_5[::4]
    for entry in val_list_bearing_5:
        train_list_bearing_5.remove(entry)

    network = ClassifierNN(input_length=1024).cuda()
    print(network)

    train_data_bearing_5 = expand_and_annotate_list_bearing_5(train_list_bearing_5, gt_list_bearing_5)
    test_data_bearing_5 = expand_and_annotate_list_bearing_5(test_list_bearing_5, gt_list_bearing_5)
    val_data_bearing_5 = expand_and_annotate_list_bearing_5(val_list_bearing_5, gt_list_bearing_5)

    roi_list_bearing_4 = load_txt_list("roi/Lager4.txt")
    gt_list_bearing_4 = load_txt_list("gt/Lager4.txt")

    train_list_bearing_4 = roi_list_bearing_4[::2]
    test_list_bearing_4 = roi_list_bearing_4[1::2]
    val_list_bearing_4 = train_list_bearing_4[::4]
    for entry in val_list_bearing_4:
        train_list_bearing_4.remove(entry)

    test_data_bearing_4 = expand_and_annotate_list_bearing_4(test_list_bearing_4, gt_list_bearing_4)
    val_data_bearing_4 = expand_and_annotate_list_bearing_4(val_list_bearing_4, gt_list_bearing_4)

    print("Loading training data bearing 5...")
    train_data_loader_bearing_5 = DataLoader(train_data_bearing_5, batch_size=1024, sensor=0, sample_length=1024)
    print("Loading validation data bearing 5...")
    val_data_loader_bearing_5 = DataLoader(val_data_bearing_5, batch_size=1024, sensor=0, sample_length=1024)
    print("Loading validation data bearing 4...")
    val_data_loader_bearing_4 = DataLoader(val_data_bearing_4, batch_size=1024, sensor=0, sample_length=1024)

    train_balance = train_data_loader_bearing_5.get_category_balance()
    weight_class_one = train_balance[0] / train_balance[1]
    
    optimizer = AdamW(network.parameters())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, weight_class_one]).cuda())

    network.load_state_dict(torch.load("classifier_model_bearing_4.p"))

    losses_bearing_5 = []
    acc_bearing_5 = []
    acc_bearing_4 = []

    for current_epoch in range(epochs):
        loss_aggregation = 0
        acc_aggregation = 0
        print(f"Epoch #{current_epoch + 1}:")

        print("Training:")
        network.train()
        for i, (input, target) in enumerate(train_data_loader_bearing_5):
            optimizer.zero_grad()

            prediction = network(input.permute(0, 2, 1))
            loss = criterion(prediction, target)
            loss_aggregation += loss.item()
            acc_aggregation += accuracy_metric(prediction, target)

            print(f"\rloss: {loss_aggregation / (i + 1):.4f}, acc: {acc_aggregation / (i + 1):.2f}", end="")

            loss.backward()
            optimizer.step()
        losses_bearing_5.append(loss_aggregation / (i + 1))
        print()
        print("Validation_bearing_5")
        loss_aggregation = 0
        acc_aggregation = 0
        network.eval()
        with torch. no_grad():
            for i, (input, target) in enumerate(val_data_loader_bearing_5):
                
                prediction = network(input.permute(0, 2, 1))
                acc_aggregation += accuracy_metric(prediction, target)

                print(f"\racc_bearing_5: {acc_aggregation / (i + 1):.2f}", end="")
        acc_bearing_5.append(acc_aggregation / (i + 1))
        print()
        print("Validation_bearing_4")
        loss_aggregation = 0
        acc_aggregation = 0
        network.eval()
        with torch. no_grad():
            for i, (input, target) in enumerate(val_data_loader_bearing_4):
                
                prediction = network(input.permute(0, 2, 1))
                acc_aggregation += accuracy_metric(prediction, target)

                print(f"\racc_bearing_4: {acc_aggregation / (i + 1):.2f}", end="")
        acc_bearing_4.append(acc_aggregation / (i + 1))
        print()    

    pickle.dump({"loss_bearing_5": losses_bearing_5, "acc_bearing_5": acc_bearing_5, "acc_bearing_4": acc_bearing_4}, open("classify_continue_training_on_5_stats.p", "wb"))
    print("Test bearing 5")
    network.eval()
    num_correct = 0
    plot_data_bearing_5 = []
    gt_begin_index = 0
    with torch. no_grad():
        for i, (data_file, category) in enumerate(test_data_bearing_5):
            if "2019_03_16__14_52_12.mat" in data_file or "2019_03_16__14_54_11.mat" in data_file:
                gt_begin_index = i
            print(f"\rProcessing {i + 1}/{len(test_data_bearing_5)}", end="")
            ts = loadmat(data_file)["Data"][..., 0, np.newaxis]

            num_chunks = ts.shape[0] // 1024
            chunks = np.array_split(ts, num_chunks)
                
            chunk_aggregator = []
            votes = []
            confidences = []
            for chunk in chunks:
                if chunk.shape[0] != 1024:
                    continue
                chunk = normalize_sample(chunk)
                chunk_aggregator.append(chunk)

                if len(chunk_aggregator) == 1024:
                    new_votes, new_confidences = predict_test_batch(chunk_aggregator, network)
                    confidences.extend(new_confidences)
                    votes.extend(new_votes)
            
            if len(chunk_aggregator) > 0:
                new_votes, new_confidences = predict_test_batch(chunk_aggregator, network)
                confidences.extend(new_confidences)
                votes.extend(new_votes)

            majority_vote, confidence = find_majority_with_confidence(votes, confidences)
            # majority_vote = max(set(votes), key = votes.count)
            # confidence_score = votes.count(majority_vote) / len(votes)
            plot_data_bearing_5.append([majority_vote, confidence, data_file.split("/")[-1].split(".")[0], category])
            if majority_vote == category:
                num_correct += 1
    print()            
    percentage_correct = num_correct / len(plot_data_bearing_5) * 100
    print(f"Precentage correct: {percentage_correct:.2f}%")

    # print(plot_data)
    plot_data_bearing_5 = np.asarray(plot_data_bearing_5)
    np.save("classify_continue_plot_data_on_bearing_5_test_5.npy", plot_data_bearing_5)

    print("Test bearing 4")
    network.eval()
    num_correct = 0
    plot_data_bearing_4 = []
    gt_begin_index = 0
    with torch. no_grad():
        for i, (data_file, category) in enumerate(test_data_bearing_4):
            if "2019_02_28__23_25_17.mat" in data_file:
                gt_begin_index = i
            print(f"\rProcessing {i + 1}/{len(test_data_bearing_4)}", end="")
            ts = loadmat(data_file)["Data"][..., 0, np.newaxis]

            num_chunks = ts.shape[0] // 1024
            chunks = np.array_split(ts, num_chunks)
                
            chunk_aggregator = []
            votes = []
            confidences = []
            for chunk in chunks:
                if chunk.shape[0] != 1024:
                    continue
                chunk = normalize_sample(chunk)
                chunk_aggregator.append(chunk)

                if len(chunk_aggregator) == 1024:
                    new_votes, new_confidences = predict_test_batch(chunk_aggregator, network)
                    confidences.extend(new_confidences)
                    votes.extend(new_votes)
            
            if len(chunk_aggregator) > 0:
                new_votes, new_confidences = predict_test_batch(chunk_aggregator, network)
                confidences.extend(new_confidences)
                votes.extend(new_votes)

            majority_vote, confidence = find_majority_with_confidence(votes, confidences)
            # majority_vote = max(set(votes), key = votes.count)
            # confidence_score = votes.count(majority_vote) / len(votes)
            plot_data_bearing_4.append([majority_vote, confidence, data_file.split("/")[-1].split(".")[0], category])
            if majority_vote == category:
                num_correct += 1
    print()            
    percentage_correct = num_correct / len(plot_data_bearing_4) * 100
    print(f"Precentage correct: {percentage_correct:.2f}%")

    # print(plot_data)
    plot_data_bearing_4 = np.asarray(plot_data_bearing_4)
    np.save("classify_continue_plot_data_on_bearing_5_test_4.npy", plot_data_bearing_4)
    # plt.figure(figsize=(16, 9), dpi=800)
    # viridis = plt.cm.get_cmap('viridis')
    # colors = ["g" if x == 0 else "r" for x in plot_data[:,0].astype(np.float)]
    # # plt.scatter(x=np.arange(len(plot_data)), y=plot_data[:,1], c=viridis(plot_data[:,1]))
    # plt.ylim(0, 1)
    # plt.scatter(x=np.arange(len(plot_data)), y=plot_data[:,1].astype(np.float), c=colors)
    # plt.xticks(np.arange(len(plot_data))[::4], plot_data[:,2][::4], rotation=25)
    # plt.vlines(gt_begin_index, 0, 1)
    # plt.hlines(y=0.5, linestyles="dashed", xmin=0, xmax=len(plot_data))
    # plt.savefig("classify_plot.png")


if __name__ == "__main__":
    main()