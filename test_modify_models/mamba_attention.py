from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
from splicing_block.MLLA1D import *

model_test = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
attention_in_channels = model_test.features[5].out_channels

class MobileNetV3_MamBa(nn.Module):
    def __init__(self):
        super(MobileNetV3_MamBa, self).__init__()

        # 加载 MobileNetV3 模型
        self.mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.mobilenet.classifier = nn.Identity()  # 删除原有 classifier 层
        # 插入自注意力模块
        self.mlla_block = MLLABlock(dim=40, input_resolution=784)

    def mamba_cal(self, x):
        # 转换 (B, C, H, W) -> (B, L, C)，其中 L = H * W
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # 现在 x 的形状为 (B, L, C)，L = H * W
        # 添加自注意力机制
        x = self.mlla_block(x)
        # 转换回 (B, C, H, W) 形状以继续经过后续的 MobileNetV3 层
        x = x.permute(0, 2, 1).view(B, C, H, W)  # 重新转换为 (B, C, H, W)

        return x

    def forward(self, x):
        # 先经过 MobileNetV3 的 features 层
        x = self.mobilenet.features[:5](x)
        # print(x.shape)
        x = self.mamba_cal(x)
        x = self.mobilenet.features[5:6](x)
        x = self.mamba_cal(x)
        x = self.mobilenet.features[6:7](x)
        x = self.mamba_cal(x)
        x = self.mobilenet.features[7:](x)

        # 全局平均池化，转换为 (batch_size, 960, 1, 1)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 展平为 (batch_size, 960)
        # x = torch.flatten(x, 1)

        return x

if __name__ == '__main__':
    model = MobileNetV3_MamBa()
    # 测试模型，传入示例输入数据
    input_tensor = torch.randn(32, 3, 224, 224)  # 示例输入，假设为224x224的RGB图像
    output = model(input_tensor)
    print(output.size())