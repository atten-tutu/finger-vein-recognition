import torch
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
import torch.nn as nn
from splicing_block.SHSA import SHSA
from test_modify_models.CA import ChannelAggregationFFN


class SHCA(nn.Module):
    def __init__(self,dim):
        super(SHCA, self).__init__()

        # 加载 MobileNetV3 模型
        # 插入自注意力模块
        self.ca = ChannelAggregationFFN(embed_dims=dim)
        self.shsa = SHSA(dim)
        self.conv = nn.Conv2d(dim,dim,1)
        # # 引入可学习的权重（alpha, beta）
        # self.alpha = nn.Parameter(torch.ones(1))  # 用于加权x1
        # self.beta = nn.Parameter(torch.ones(1))  # 用于加权x2
        #
        # # 引入可学习的缩放因子gamma
        # self.gamma = nn.Parameter(torch.ones(1))  # 用于缩放残差连接

    # def forward(self, x):
    #     res = x
    #     x1 = self.ca(x)
    #     x2 = self.shsa(x)
    #
    #     # 用可学习的权重对两个输出进行加权
    #     weighted_x1 = self.alpha * x1
    #     weighted_x2 = self.beta * x2
    #
    #     # 将加权后的特征图相加并通过卷积
    #     x = self.conv(weighted_x1 + weighted_x2)
    #
    #     # 使用可学习的缩放因子对残差进行调整
    #     return self.gamma * x + res
    def forward(self, x):
        res = x
        x1=self.ca(x)
        x2=self.shsa(x)
        x = self.conv(x1+x2)
        return x+res

if __name__ == '__main__':
    model =SHCA(40)
    # 测试模型，传入示例输入数据
    input_tensor = torch.randn(1, 40, 28, 28)  # 示例输入，假设为224x224的RGB图像
    output = model(input_tensor)
    print(output.size())