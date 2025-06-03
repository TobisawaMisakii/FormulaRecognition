from preprocess.preprocess_data import FormulaSample
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import os
import pandas as pd

def extract_bert_features(df, batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 从df构造FormulaSample列表，只取必要字段
    samples = []
    for _, row in df.iterrows():
        sample = FormulaSample(
            no=str(row["no"]),
            image=None,  # 这里只提取文本特征，图像暂时不需要
            text=row["text"],
            label_id=int(row["label_id"]),
            label_name=row["label_name"],
            is_handwritten=bool(row["is_handwritten"])
        )
        samples.append(sample)

    model = BertModel.from_pretrained("/home/cpy/prml-cls/HandWritten-MathFormula-Recognition/models/bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(
        "/home/cpy/prml-cls/HandWritten-MathFormula-Recognition/models/bert-base-chinese")
    model.to(device)
    model.eval()

    texts = [sample.text for sample in samples]
    features = []

    # 分批处理（避免内存溢出）
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # 分词并转换为模型输入
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512  # BERT最大长度限制
        ).to(device)
        # 提取特征
        with torch.no_grad():
            outputs = model(**inputs)
            # 取[CLS]标记的特征（或平均池化）
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(batch_features)

    return np.vstack(features)


if __name__ == "__main__":
    data_root = "../data/processed"
    metadata_path = os.path.join(data_root, "metadata.csv")
    df = pd.read_csv(metadata_path)

    bert_features = extract_bert_features(df)
    print("BERT特征维度:", bert_features.shape)  # (*, 768)
    print("BERT特征示例:", bert_features[0])  # 打印第一个样本的特征向量