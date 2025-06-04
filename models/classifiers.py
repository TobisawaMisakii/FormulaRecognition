# 分类模型的训练与评估
# KNN、朴素贝叶斯、MLP、SVM、CNN

import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from features.dim_reduce import extract_features, dimensionality_reduction


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * (input_dim // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def filter_small_classes(X, y, min_count=2):
    counter = Counter(y)
    keep_classes = [cls for cls, count in counter.items() if count >= min_count]

    indices = [i for i, label in enumerate(y) if label in keep_classes]
    return X[indices], y[indices]


def train_and_evaluate_classifier(X, y, train_size, method, n_components, classifier_type, **kwargs):
    if not os.path.exists(f'cls_report/{classifier_type}'):
        os.makedirs(f'cls_report/{classifier_type}')
    if os.path.exists(f'cls_report/{classifier_type}/{method}_{n_components}d.txt'):
        os.remove(f'cls_report/{classifier_type}/{method}_{n_components}d.txt')

    report_file = f'cls_report/{classifier_type}/{method}_{n_components}d.txt'

    if classifier_type == 'knn':
        model = KNeighborsClassifier(**kwargs)
    elif classifier_type == 'naive_bayes':
        model = GaussianNB(**kwargs)
    elif classifier_type == 'svm':
        model = SVC(**kwargs)
    elif classifier_type == 'mlp':
        model = MLPClassifier(**kwargs)
    elif classifier_type == 'cnn':
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        model = SimpleCNN(input_dim=input_dim, num_classes=num_classes)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Split data into training and testing sets
    X_filtered, y_filtered = filter_small_classes(X, y, min_count=2)    # 过滤掉样本数小于2的类别
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered,
                                                        train_size=train_size, stratify=y_filtered)

    if classifier_type == 'cnn':
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 1e-3))

        # Training loop
        model.train()
        for epoch in range(kwargs.get('epochs', 40)):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{kwargs.get("epochs", 40)}], Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
            with open(report_file, 'a') as f:
                f.write(f'{classifier_type.upper()} Test Accuracy: {accuracy:.4f}\n')
                f.write(classification_report(y_test_tensor.cpu(), predicted.cpu(), zero_division=0))

    else:
        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{classifier_type.upper()} Test Accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred, zero_division=0))
        with open(report_file, 'a') as f:
            f.write(f'{classifier_type.upper()} Test Accuracy: {accuracy:.4f}\n')
            f.write(classification_report(y_test, y_pred, zero_division=0))

    return accuracy


if __name__ == "__main__":
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))

    features = extract_features(df, data_root)
    labels = df['label_id'].values
    print("标签维度:", labels.shape)

    dim_reduce_method = 'Autoencoder'  # 可以选择 'PCA', 'LDA', 'Autoencoder'
    n_components = 128  # LDA 降维后最多 14 维， PCA 降维后最多 259 维， Autoencoder 可以选择 128

    features_train = dimensionality_reduction(features, labels=labels, method=dim_reduce_method,
                                              n_components=n_components)

    # train and evaluate classifiers
    classifiers = {
        'knn': {'n_neighbors': 5},
        'naive_bayes': {},
        'svm': {'kernel': 'linear', 'C': 1.0},
        'mlp': {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42},
        'cnn': {'epochs': 100, 'lr': 0.001}
    }
    train_size = 0.8  # 80% training data
    for clf_name, clf_params in classifiers.items():
        print(f"Training {clf_name.upper()} classifier...")
        accuracy = train_and_evaluate_classifier(X=features_train, y=labels,
                                                 train_size=train_size, classifier_type=clf_name,
                                                 method=dim_reduce_method,
                                                 n_components=n_components, **clf_params)
        print(f"{clf_name.upper()} Classifier Accuracy: {accuracy:.4f}\n")
