# PCA/LDA/Autoencoder 降维

from features.image_features import extract_image_features_from_npy
from features.text_features import extract_bert_features
from utils.visualization import visualize_2d_scatter, visualize_3d_interactive
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

def reduce_with_autoencoder(features, n_components=2, epochs=100, batch_size=64, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = features.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=n_components).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        reduced = model.encode(features.to(device)).cpu().numpy()
    return reduced

def extract_features(df, data_root, batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))
    batch_size = 8

    img_features = extract_image_features_from_npy(df, data_root, batch_size)
    img_features = torch.tensor(img_features, dtype=torch.float32).to(device)
    print("图像特征维度:", img_features.shape)
    bert_features = extract_bert_features(df, batch_size)
    bert_features = torch.tensor(bert_features, dtype=torch.float32).to(device)
    print("BERT特征维度:", bert_features.shape)

    # 合并特征，文本在前，图像在后
    features = torch.cat((bert_features, img_features), dim=1)
    print("合并特征维度:", features.shape)

    return features

def dimensionality_reduction(features, labels=None, method='PCA', n_components=2):
    # 特征降维，可以选择降到 2/3维（可视化散点图） / 1024维度（用于训练）
    # 这里可以使用 PCA / LDA / Autoencoder 等方法进行降维
    if method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features.cpu().numpy())
    elif method == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=n_components)
        reduced_features = lda.fit_transform(features.cpu().numpy(), labels)
    elif method == 'Autoencoder':
        reduced_features = reduce_with_autoencoder(features, n_components)
    else:
        raise ValueError("Unsupported dimensionality reduction method: {}".format(method))
    return reduced_features


if __name__ == "__main__":
    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))

    features = extract_features(df, data_root)
    labels = df['label_id'].values
    print("特征维度:", features.shape)

    # reduced_features = dimensionality_reduction(features, method='PCA', n_components=2)
    # visualize_2d_scatter(reduced_features, reduce_method='PCA', labels=df['label_id'].values, title="PCA 2D Scatter Plot")
    #
    # reduced_features = dimensionality_reduction(features, labels=labels, method='LDA', n_components=2)
    # visualize_2d_scatter(reduced_features, reduce_method='LDA', labels=df['label_id'].values, title="LDA 2D Scatter Plot")
    #
    # reduced_features = dimensionality_reduction(features, method='Autoencoder', n_components=2)
    # visualize_2d_scatter(reduced_features, reduce_method='Autoencoder', labels=df['label_id'].values, title="Autoencoder 2D Scatter Plot")

    reduced_features = dimensionality_reduction(features, method='PCA', n_components=3)
    visualize_3d_interactive(reduced_features, reduce_method='PCA', labels=df['label_id'].values, title="PCA 3D Interactive Plot")

    reduced_features = dimensionality_reduction(features, labels=labels, method='LDA', n_components=3)
    visualize_3d_interactive(reduced_features, reduce_method='LDA', labels=df['label_id'].values, title="LDA 3D Interactive Plot")

    reduced_features = dimensionality_reduction(features, method='Autoencoder', n_components=3)
    visualize_3d_interactive(reduced_features, reduce_method='Autoencoder', labels=df['label_id'].values, title="Autoencoder 3D Interactive Plot")