# 数据预处理
# 缺失值处理、归一化等

# preprocess/preprocess_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
import shutil

formula_type = {1: "Bernoulli", 2: "Beta", 3: "Binomial", 4: "Gamma", 5: "GMM",
                6: "Even", 7: "Chi-square", 8: "Rice", 9: "PowerLow", 10: "Pareto",
                11: "Poisson", 12: "Normal", 13: "Exponential", 14: "Hypergeometric", 15: "Dirichlet"}

class FormulaSample:
    def __init__(self, image, text, no, label_id, label_name, is_handwritten):
        self.image = image            # 图像数据（numpy.ndarray）
        self.text = text              # 描述性文本（字符串）
        self.no = no                  # 公式编号（唯一）
        self.label_id = label_id      # 数值标签（1~15）
        self.label_name = label_name  # 分布名称（如 "Poisson"）
        self.is_handwritten = is_handwritten  # 是否手写公式

    def __repr__(self):
        return f"<FormulaSample {self.label_id}-{self.label_name}>"

def read_text_with_multiple_encodings(file_path):
    """尝试用 utf-8、gbk、utf-16le 打开文本"""
    encodings = ['utf-8', 'gbk', 'utf-16le']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    print(f"读取失败：{file_path}（尝试了 utf-8, gbk, utf-16le）")
    return None


def load_raw_data(data_dir):
    samples = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # 提取标签信息
        formula_id = int(folder.split('_')[0])
        formula_name = formula_type[formula_id]
        formula_no = int(folder.split('_')[1].strip("y"))
        image = None
        text = None
        # 检查是否为手写公式
        is_handwritten = folder.endswith('y')
        # 遍历该目录内的文件
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.txt'):
                text = read_text_with_multiple_encodings(file_path)
            elif file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.PNG') or file.endswith('.JPG'):
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    image = img

        # 如果图像和文本都成功读取
        if image is not None and text is not None:
            sample = FormulaSample(no=formula_no, image=image, text=text, label_id=formula_id, label_name=formula_name, is_handwritten=is_handwritten)
            samples.append(sample)
        else:
            print(f"缺失图像或文本，跳过：{folder}")

    return samples

def preprocess_data(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir)

    samples = load_raw_data(input_dir)
    metadata = []

    for sample in samples:
        # 图像归一化
        img = sample.image.astype(np.float32) / 255.0

        # 保存图像为 .npy 文件
        img_filename = f"{sample.no}_{sample.label_name}.npy"
        img_path = os.path.join(img_dir, img_filename)
        np.save(img_path, img)

        # 收集元信息
        metadata.append({
            "no": sample.no,
            "label_id": sample.label_id,
            "label_name": sample.label_name,
            "is_handwritten": sample.is_handwritten,
            "image_path": os.path.relpath(img_path, output_dir),
            "text": sample.text
        })

    # 保存所有元信息为 metadata.csv
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"共处理并保存样本：{len(samples)} 条。输出目录：{output_dir}")


if __name__ == "__main__":
    preprocess_data("../data/raw", "../data/processed")
