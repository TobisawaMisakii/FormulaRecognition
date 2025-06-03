# extract_image_features_from_npy.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class NpyImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.root_dir, row["image_path"])
        image = np.load(npy_path)  # shape: (H, W, 3), already normalized [0, 1]

        if self.transform:
            image = self.transform(image)
        return image

def extract_image_features_from_npy(df, data_root, batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    使用 ResNet 提取图像特征，输入为已保存的 .npy 图像数组
    :param df: 包含 image_path 列的 DataFrame
    :param data_root: .npy 文件所在的根目录（如 processed/）
    :return: np.ndarray, shape = (N, 2048)
    """

    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W), 并确保是 float32
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = NpyImageDataset(df, root_dir=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 加载 ResNet50 并去掉最后一层
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])  # 去掉最后的 fc 层
    model.to(device)
    model.eval()

    features = []

    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)  # (B, 2048, 1, 1)
            out = out.view(out.size(0), -1)  # (B, 2048)
            features.append(out.cpu().numpy())

    return np.vstack(features)

# ✅ 示例调用
if __name__ == "__main__":
    import pandas as pd

    data_root = "../data/processed"
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))

    img_features = extract_image_features_from_npy(df, data_root)
    print("图像特征维度:", img_features.shape)
