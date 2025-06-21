from torchvision import models
# from torchvision.models import MobileNet_V3_Large_Weights
from splicing_block.selfattention import *

# model_test = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
# attention_in_channels = model_test.features[5].out_channels

class MobileNetV3_Attention(nn.Module):
    def __init__(self):
        super(MobileNetV3_Attention, self).__init__()

        # 加载 MobileNetV3 模型
        self.mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.mobilenet.classifier = nn.Identity()  # 删除原有 classifier 层
        # 获取插入注意力机制的层的输出通道数
        attention_in_channels = self.mobilenet.features[5].out_channels
        # 插入自注意力模块
        self.self_attention = SelfAttention(attention_in_channels)

    def forward(self, x):
        # 先经过 MobileNetV3 的前 6 层
        x = self.mobilenet.features[:6](x)
        # 添加自注意力机制
        x = self.self_attention(x)
        # 继续经过剩下的 MobileNetV3 的层
        x = self.mobilenet.features[6:](x)
        # 全局平均池化，转换为 (batch_size, 960, 1, 1)
        x = nn.functional.adaptive_avg_pool2d(x, (1,1))
        # 展平为 (batch_size, 960)
        x = torch.flatten(x, 1)
        return x