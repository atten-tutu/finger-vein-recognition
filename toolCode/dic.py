import torch
from torch.utils.data import Dataset, DataLoader

# 定义自定义Dataset
class DictDataset(Dataset):
    def __init__(self, data_dict):
        # 将字典的键和值分离，并转为适合PyTorch的数据结构
        self.targets = list(data_dict.keys())
        self.features = list(data_dict.values())

    def __len__(self):
        # 数据的长度
        return len(self.targets)

    def __getitem__(self, idx):
        # 获取单个样本
        return self.targets[idx], self.features[idx]