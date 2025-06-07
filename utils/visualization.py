import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import os


def visualize_2d_scatter(features_2d, reduce_method, labels, title="2D Feature Scatter Plot"):
    """
    features: torch.Tensor or numpy.ndarray，shape=(N, D)
    labels: list or np.array，shape=(N,)
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title(title)
    plt.xlabel(reduce_method + " Component 1")
    plt.ylabel(reduce_method + " Component 2")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"/home/cpy/prml-cls/HandWritten-MathFormula-Recognition/figures/{reduce_method}_2d_scatter_plot.png")



def visualize_3d_interactive(features, reduce_method, labels, title="Interactive 3D Feature Visualization"):
    """
    使用 plotly 可视化 3D 特征
    :param features: np.ndarray of shape (N, 3)
    :param labels: array-like of shape (N,)
    :param title: 图标题
    """
    df = pd.DataFrame(features, columns=["x", "y", "z"])
    df['label'] = labels

    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color=df['label'].astype(str),
        title=title,
        opacity=0.8,
        height=1200,
        width=1200
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text='Class Label')
    # fig.show()
    fig.write_html(f"/home/cpy/prml-cls/HandWritten-MathFormula-Recognition/figures/{reduce_method}_3d.html")


def plot_train_size_vs_accuracy(csv_path, title=None, save_path=None):
    """
    读取稳定性csv（包含train_size和accuracy列），画出train_size与accuracy的折线图
    """
    df = pd.read_csv(csv_path)

    # 过滤掉最后一行 mean±std
    df_numeric = df[pd.to_numeric(df['train_size'], errors='coerce').notnull()].copy()
    df_numeric['train_size'] = df_numeric['train_size'].astype(float)
    df_numeric['accuracy'] = df_numeric['accuracy'].astype(float)

    plt.figure(figsize=(8, 6))
    plt.plot(df_numeric['train_size'], df_numeric['accuracy'], marker='o', linestyle='-')
    plt.title(title or "Train Size vs Accuracy")
    plt.xlabel("Train Size")
    plt.ylabel("Accuracy")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()