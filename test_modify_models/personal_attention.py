import torch
import torch.nn as nn
from splicing_block.MLLA1D import *

class personal_attention(nn.Module):
    def __init__(self):
        super(personal_attention, self).__init__()

        self.mlla_block = MLLABlock(dim=960, input_resolution=1)

        self.conv1x1 = nn.Conv2d(in_channels=960, out_channels=640, kernel_size=1)

    def mamba_cal(self, x):

        # 转换 (B, C, H, W) -> (B, L, C)，其中 L = H * W
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # 现在 x 的形状为 (B, L, C)，L = H * W
        # 添加自注意力机制
        x = self.mlla_block(x)
        # 转换回 (B, C, H, W) 形状以继续经过后续的层
        x = x.permute(0, 2, 1).view(B, C, H, W)  # 重新转换为 (B, C, H, W)
        return x

    def forward(self, x):
        for _ in range(3):
            x = self.mamba_cal(x)
        # print(x.shape)
        x = self.conv1x1(x)
        x = torch.flatten(x, 1)
        return x

if __name__ == '__main__':
    model = personal_attention()
    # 测试模型，传入示例输入数据
    input_tensor = torch.randn(32, 960, 1, 1)  # 示例输入，假设为224x224的RGB图像
    output = model(input_tensor)
    print(output.shape)
