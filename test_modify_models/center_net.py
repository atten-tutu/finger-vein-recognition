import torch.nn as nn
import torch

class CenterLossNet(nn.Module):
    def __init__(self, cls_num, feature_dim):
        super(CenterLossNet, self).__init__()
        self.centers = nn.Parameter(torch.randn(cls_num, feature_dim))

    def forward(self, features, labels, reduction='mean'):
        # 特征向量归一化
        _features = nn.functional.normalize(features)

        centers_batch = self.centers.index_select(dim=0, index=labels.long())
        # 根据论文《A Discriminative Feature Learning Approach for Deep Face Recognition》修改如下
        if reduction == 'sum':  # 返回loss的和
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2 / len(features)
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))