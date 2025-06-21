import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        # s = 16
        super(CosFaceLoss, self).__init__()
        self.s = s  # 缩放因子
        self.m = m  # 边界（余弦间隔）
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # 正则化权重和输入特征
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # 对正类添加余弦间隔
        target_logit = cosine - self.m

        # 构建输出 logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * target_logit + (1.0 - one_hot) * cosine
        # 进行缩放
        output = output * self.s

        loss = F.cross_entropy(output, label)
        return loss