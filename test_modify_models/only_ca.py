import torch
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
import torch.nn as nn
from splicing_block.SHSA import SHSA
from test_modify_models.special import SEW_MONSTER
from test_modify_models.ema import EMA
from test_modify_models.CA import ChannelAggregationFFN

class only_ca(nn.Module):
    def __init__(self):
        super(only_ca, self).__init__()

        # 加载 MobileNetV3 模型
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)

        self.mobilenet.classifier = nn.Identity()  # 删除原有 classifier 层

        # 插入自注意力模块
        self.sm_40 = ChannelAggregationFFN(40)
        self.sm_80 = ChannelAggregationFFN(80)
        self.sm_112 = ChannelAggregationFFN(160)
        # self.sm = SEW_MONSTER(960)
        # self.ema = EMA(960)
    def forward(self, x):
        # 先经过 MobileNetV3 的 features 层

        x = self.mobilenet.features[:7](x)
        x = self.sm_40(x)
        x = self.mobilenet.features[7:11](x)

        x = self.sm_80(x)
        x = self.mobilenet.features[11:16](x)

        x = self.sm_112(x)
        x = self.mobilenet.features[16:](x)
        # x = self.mobilenet.features(x)
        # x = self.ema(x)
        # 全局平均池化，转换为 (batch_size, 960, 1, 1)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 展平为 (batch_size, 960)
        x = torch.flatten(x, 1)


        return x

if __name__ == '__main__':
    model =only_ca()
    # 测试模型，传入示例输入数据
    input_tensor = torch.randn(1, 3, 224, 224)  # 示例输入，假设为224x224的RGB图像
    output = model(input_tensor)
    print(output.size())