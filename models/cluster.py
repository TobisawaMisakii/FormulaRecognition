import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from features.dim_reduce import extract_features
from features.dim_reduce import dimensionality_reduction


def cluster_accuracy(y_true, y_pred):
    """
    使用 Hungarian 算法对齐标签计算聚类准确率（ACC）
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        if y_pred[i] >= 0:
            w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(*ind)]) / len(y_pred)
    return acc


def run_clustering(X, y_true, method="kmeans", n_clusters=15):
    print("X shape:", X.shape)
    print("X min:", X.min(axis=0))
    print("X max:", X.max(axis=0))
    print("X mean:", X.mean(axis=0))
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "dbscan":
        model = DBSCAN(eps=3, min_samples=5)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported method: {method}")

    y_pred = model.fit_predict(X)

    # 对于 DBSCAN 的 -1 噪声点，过滤掉
    mask = y_pred != -1 if method == "dbscan" else np.ones(len(y_pred), dtype=bool)

    print("mask sum:", mask.sum())
    print("y_true[mask] shape:", np.array(y_true)[mask].shape)
    print("y_pred[mask] shape:", y_pred[mask].shape)

    acc = cluster_accuracy(np.array(y_true)[mask], y_pred[mask])
    nmi = normalized_mutual_info_score(np.array(y_true)[mask], y_pred[mask])
    return y_pred, acc, nmi


def analyze_cluster_composition(y_pred, y_true, label_map):
    """
    输出每个聚类簇内的真实类别组成（Top 3）
    """
    clusters = defaultdict(list)
    for pred, true in zip(y_pred, y_true):
        if pred != -1:
            clusters[pred].append(label_map[true])
    print("\n📊 聚类簇组成分析：")
    for cid, members in sorted(clusters.items()):
        top = Counter(members).most_common(3)
        print(f"Cluster {cid}: {top}")


def plot_tsne(X, y, title="t-SNE cluster visualization", save_path=None):
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=10)
    plt.title(title)
    if save_path:
        save_path = os.path.join("cluster_report", save_path)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved t-SNE figure to {save_path}")
    else:
        plt.show()


def plot_hierarchical_dendrogram(X, method='ward'):
    Z = linkage(X, method=method)
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='level', p=10)
    plt.title("hierachical cluster - dendrogram")
    plt.xlabel("sample")
    plt.ylabel("distance")
    plt.tight_layout()
    # plt.show()
    plt.savefig("cluster_report/hierarchical_dendrogram.png")


def main_clustering():
    # 加载数据
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))
    label_ids = df['label_id'].values
    formula_map = {
        1: "Bernoulli", 2: "Beta", 3: "Binomial", 4: "Gamma", 5: "GMM",
        6: "Even", 7: "Chi-square", 8: "Rice", 9: "PowerLow", 10: "Pareto",
        11: "Poisson", 12: "Normal", 13: "Exponential", 14: "Hypergeometric", 15: "Dirichlet"
    }

    # 特征提取 + LDA 降维
    features, labels = extract_features(df, data_root=data_root)
    X_reduced = dimensionality_reduction(features, labels, method="LDA", n_components=8)

    # 运行不同聚类算法
    for method in ["kmeans", "dbscan", "hierarchical"]:
        print(f"\n=== Running Clustering: {method.upper()} ===")
        y_pred, acc, nmi = run_clustering(X_reduced, labels, method=method, n_clusters=5)
        print(f"[{method.upper()}] ACC: {acc:.4f} | NMI: {nmi:.4f}")

        analyze_cluster_composition(y_pred, labels, formula_map)
        plot_tsne(X_reduced, y_pred, title=f"t-SNE cluster result ({method})",
                  save_path=f"cluster_{method}_tsne.png")

        if method == "hierarchical":
            plot_hierarchical_dendrogram(X_reduced)

    # 额外分析：手写 vs 打印的聚类效果
    print("\n=== 手写 vs 打印聚类分析 ===")
    df_hand = df[df['is_handwritten'] == True]
    df_print = df[df['is_handwritten'] == False]
    for name, subset in [("Handwritten", df_hand), ("Printed", df_print)]:
        print(f"\n>> 聚类分析：{name}")
        X_sub, y_sub = extract_features(subset, data_root=data_root)
        X_sub_reduced = dimensionality_reduction(X_sub, y_sub, method="PCA", n_components=30)
        y_pred, acc, nmi = run_clustering(X_sub_reduced, y_sub, method="kmeans", n_clusters=15)
        print(f"{name} - KMeans ACC: {acc:.4f} | NMI: {nmi:.4f}")
        plot_tsne(X_sub_reduced, y_pred, title=f"{name} cluster t-SNE", save_path=f"{name.lower()}_tsne.png")


if __name__ == "__main__":
    main_clustering()
