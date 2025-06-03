import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd


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