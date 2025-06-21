import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # 输入维度
        self.in_dim = in_dim
        # Query, Key, Value 的线性变换
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # Softmax 用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)
        # 归一化因子
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 获取输入的形状
        batch_size, C, width, height = x.size()
        # 对输入进行 Query、Key 和 Value 的线性变换
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # [B, N, C]
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # [B, C, N]
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # [B, C, N]

        # 计算 QK^T 并通过 softmax 得到注意力矩阵
        attention = self.softmax(torch.bmm(proj_query, proj_key))  # [B, N, N]

        # 计算加权的 Value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, width, height)  # 恢复原始维度

        # 输出是原始输入 + gamma 调整的注意力输出
        out = self.gamma * out + x
        return out
