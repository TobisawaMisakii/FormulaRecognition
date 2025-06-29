# 分类模型的训练与评估
# KNN、朴素贝叶斯、MLP、SVM、CNN

import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

from features.dim_reduce import extract_features, dimensionality_reduction
from utils.visualization import plot_train_size_vs_accuracy, plot_handwritten_vs_printed_accuracy

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
    classes_in_test = np.unique(y_test)

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

            # CNN 输出 softmax 概率
            probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
            num_classes = probs.shape[1]
            y_test_np = y_test_tensor.cpu().numpy()

            y_test_bin = label_binarize(y_test_np, classes=np.arange(num_classes))
            valid_cols = y_test_bin.sum(axis=0) > 0

            with open(report_file, 'a') as f:
                if valid_cols.sum() < 2:
                    f.write("AUC 无法计算（测试集中可用类别不足）\n")
                else:
                    macro_auc = roc_auc_score(y_test_bin[:, valid_cols], probs[:, valid_cols],
                                              average='macro', multi_class='ovr')
                    weighted_auc = roc_auc_score(y_test_bin[:, valid_cols], probs[:, valid_cols],
                                                 average='weighted', multi_class='ovr')
                    per_class_auc = roc_auc_score(y_test_bin[:, valid_cols], probs[:, valid_cols],
                                                  average=None, multi_class='ovr')

                    f.write(f"AUC (macro): {macro_auc:.4f}\n")
                    f.write(f"AUC (weighted): {weighted_auc:.4f}\n")
                    f.write(f"AUC (per-class): {np.round(per_class_auc, 4).tolist()}\n")

    else:
        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{classifier_type.upper()} Test Accuracy: {accuracy:.4f}')
        # print(classification_report(y_test, y_pred, zero_division=0))
        with open(report_file, 'a') as f:
            f.write(f'{classifier_type.upper()} Test Accuracy: {accuracy:.4f}\n')
            f.write(classification_report(y_test, y_pred, zero_division=0))

        # 多类 AUC
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_filtered))
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        if y_pred_proba is not None:
            valid_cols = y_test_binarized.sum(axis=0) > 0
            with open(report_file, 'a') as f:
                if valid_cols.sum() < 2:
                    f.write("AUC 无法计算（测试集中可用类别不足）\n")
                else:
                    macro_auc = roc_auc_score(y_test_binarized[:, valid_cols], y_pred_proba[:, valid_cols],
                                              average='macro', multi_class='ovr')
                    weighted_auc = roc_auc_score(y_test_binarized[:, valid_cols], y_pred_proba[:, valid_cols],
                                                 average='weighted', multi_class='ovr')
                    per_class_auc = roc_auc_score(y_test_binarized[:, valid_cols], y_pred_proba[:, valid_cols],
                                                  average=None, multi_class='ovr')

                    f.write(f"AUC (macro): {macro_auc:.4f}\n")
                    f.write(f"AUC (weighted): {weighted_auc:.4f}\n")
                    f.write(f"AUC (per-class): {np.round(per_class_auc, 4).tolist()}\n")
        else:
            with open(report_file, 'a') as f:
                f.write("AUC: -1 (不支持概率预测的分类器)\n")

    return accuracy


def evaluate_stability(X, y, method, n_components, classifier_type,
                       train_sizes, metric='accuracy', **kwargs):
    """
    多次训练并评估模型稳定性（准确率或AUC波动）
    :return: mean, std
    """
    scores = []
    for train_size in train_sizes:
        print(f"Evaluating {classifier_type.upper()} with train/test size "
              f"{train_size:.2f}/{1.0-train_size:.2f} on {method} {n_components}D features...")
        score = train_and_evaluate_classifier(X, y, train_size=train_size,
                                              method=method, n_components=n_components,
                                              classifier_type=classifier_type, **kwargs)
        scores.append(score)

    scores = np.array(scores)
    print(f"[{classifier_type.upper()} on {method} {n_components}D]: "
          f"Mean {metric}: {scores.mean():.4f} | Std: {scores.std():.4f}")
    return scores


def main_accuracy():
    """
    主函数1：加载数据，提取特征，降维，训练和评估分类器
    :return:
    """
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))

    features = extract_features(df, data_root)
    labels = df['label_id'].values
    print("标签维度:", labels.shape)

    dim_reduce_method = 'LDA'  # 可以选择 'PCA', 'LDA', 'Autoencoder'
    n_components = 14  # LDA 降维后最多 14 维， PCA 降维后最多 259 维， Autoencoder 可以选择 128

    features_train = dimensionality_reduction(features, labels=labels, method=dim_reduce_method,
                                              n_components=n_components)

    # train and evaluate classifiers
    classifiers = {
        'knn': {'n_neighbors': 5},
        'naive_bayes': {},
        'svm': {'kernel': 'linear', 'C': 1.0},
        'mlp': {'hidden_layer_sizes': (100,), 'max_iter': 800, 'random_state': 42},
        'cnn': {'epochs': 100, 'lr': 0.001}
    }
    train_size = 0.8  # 80% training data
    for clf_name, clf_params in classifiers.items():
        print(f"Training {clf_name.upper()} classifier...")
        accuracy = train_and_evaluate_classifier(X=features_train, y=labels,
                                                 train_size=train_size, classifier_type=clf_name,
                                                 method=dim_reduce_method,
                                                 n_components=n_components, **clf_params)


def main_stability():
    """
    主函数2：加载数据，提取特征，降维，评估分类器稳定性
    :return:
    """
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))

    features = extract_features(df, data_root)
    labels = df['label_id'].values
    print("标签维度:", labels.shape)

    settings = [
        {'method': 'PCA', 'n_components': 64},
        {'method': 'LDA', 'n_components': 14},
        {'method': 'Autoencoder', 'n_components': 128}
    ]

    classifiers = {
        'knn': {'n_neighbors': 5},
        'naive_bayes': {},
        'svm': {'kernel': 'linear', 'C': 1.0},
        'mlp': {'hidden_layer_sizes': (100,), 'max_iter': 800, 'random_state': 42},
        'cnn': {'epochs': 100, 'lr': 0.001}
    }

    train_sizes = np.linspace(0.35, 0.85, 30)  # 训练集大小

    for setting in settings:
        dim_reduce_method = setting['method']
        n_components = setting['n_components']

        print(f"\nRunning setting: {dim_reduce_method}_{n_components}d")

        features_train = dimensionality_reduction(features, labels=labels,
                                                  method=dim_reduce_method,
                                                  n_components=n_components)

        for clf_name, clf_params in classifiers.items():
            print(f"  Evaluating stability of {clf_name.upper()}...")
            scores = evaluate_stability(X=features_train, y=labels,
                                        method=dim_reduce_method,
                                        n_components=n_components,
                                        classifier_type=clf_name,
                                        train_sizes=train_sizes,
                                        **clf_params)

            # Save results
            result_dir = f'cls_report/{clf_name}'
            os.makedirs(result_dir, exist_ok=True)
            stability_file = os.path.join(result_dir, f'{dim_reduce_method}_{n_components}d_stability.csv')

            if os.path.exists(stability_file):
                os.remove(stability_file)

            train_sizes_str = [f"{size:.2f}" for size in train_sizes]
            stability_df = pd.DataFrame({
                'train_size': train_sizes_str,
                'accuracy': scores
            })
            mean_acc = np.mean(scores)
            std_acc = np.std(scores)
            summary_row = pd.DataFrame([{
                'train_size': 'mean±std',
                'accuracy': f'{mean_acc:.4f}±{std_acc:.4f}'
            }])
            stability_df = pd.concat([stability_df, summary_row], ignore_index=True)
            stability_df.to_csv(stability_file, index=False)

            plot_path = os.path.join(result_dir, f"{dim_reduce_method}_{n_components}d_stability.png")
            plot_train_size_vs_accuracy(csv_path=stability_file,
                                        title=f"{clf_name.upper()} - {dim_reduce_method} {n_components}D",
                                        save_path=plot_path)


def main_handwritten_impact():
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))
    df_handwritten = df[df['is_handwritten'] == True].reset_index(drop=True)
    df_printed = df[df['is_handwritten'] == False].reset_index(drop=True)

    # 特征提取
    X_hand, y_hand = extract_features(df_handwritten)
    X_print, y_print = extract_features(df_printed)

    # 降维配置
    dim_reduce_method = 'LDA'  # 'PCA', 'LDA', 'Autoencoder'
    n_components = 8

    # 分类器设置
    classifiers = {
        'knn': {'n_neighbors': 5},
        'naive_bayes': {},
        'svm': {'kernel': 'linear', 'C': 1.0},
        'mlp': {'hidden_layer_sizes': (100,), 'max_iter': 800, 'random_state': 42},
        'cnn': {'epochs': 100, 'lr': 0.001}
    }

    train_sizes = np.linspace(0.35, 0.85, 30)

    # 对每个分类器进行手写 vs 打印的对比
    for clf_name, clf_params in classifiers.items():
        print(f"\nEvaluating classifier: {clf_name.upper()}")

        # 降维（分别处理）
        X_hand_reduced = dimensionality_reduction(X_hand, labels=y_hand,
                                                  method=dim_reduce_method,
                                                  n_components=n_components)
        X_print_reduced = dimensionality_reduction(X_print, labels=y_print,
                                                   method=dim_reduce_method,
                                                   n_components=n_components)

        print("  Evaluating on handwritten samples...")
        scores_hand = evaluate_stability(X=X_hand_reduced, y=df_handwritten['label_id'],
                                         method=dim_reduce_method,
                                         n_components=n_components,
                                         classifier_type=clf_name,
                                         train_sizes=train_sizes, **clf_params)

        print("  Evaluating on printed samples...")
        scores_print = evaluate_stability(X=X_print_reduced, y=df_printed['label_id'],
                                          method=dim_reduce_method,
                                          n_components=n_components,
                                          classifier_type=clf_name,
                                          train_sizes=train_sizes, **clf_params)

        # 保存 CSV 结果
        result_dir = f'cls_report/{clf_name}'
        os.makedirs(result_dir, exist_ok=True)
        stability_file = os.path.join(result_dir, f'{dim_reduce_method}_{n_components}d_handwritten_impact.csv')
        if os.path.exists(stability_file):
            os.remove(stability_file)

        train_sizes_str = [f"{size:.2f}" for size in train_sizes]
        stability_df = pd.DataFrame({
            'train_size': train_sizes_str,
            'handwritten_accuracy': scores_hand,
            'printed_accuracy': scores_print
        })

        # 添加平均±方差
        mean_hand = np.mean(scores_hand)
        std_hand = np.std(scores_hand)
        mean_print = np.mean(scores_print)
        std_print = np.std(scores_print)
        summary_row = pd.DataFrame([{
            'train_size': 'mean±std',
            'handwritten_accuracy': f'{mean_hand:.4f}±{std_hand:.4f}',
            'printed_accuracy': f'{mean_print:.4f}±{std_print:.4f}'
        }])
        stability_df = pd.concat([stability_df, summary_row], ignore_index=True)
        stability_df.to_csv(stability_file, index=False)

        # 画图
        plot_path = os.path.join(result_dir, f'{dim_reduce_method}_{n_components}d_handwritten_impact.png')
        plot_handwritten_vs_printed_accuracy(csv_path=stability_file,
                                             title=f"{clf_name.upper()} - {dim_reduce_method} {n_components}D",
                                             save_path=plot_path)


if __name__ == "__main__":
    # main_accuracy()  # 训练和评估分类器
    # main_stability()  # 评估分类器稳定性
    main_handwritten_impact()  # 手写和印刷体的影响