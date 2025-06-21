import torch
import torch.nn as nn
from splicing_block.SHSA import *

class personal_shsa(nn.Module):
    def __init__(self):
        super(personal_shsa, self).__init__()

        self.shsa_block_960 = SHSA(960)
        self.shsa_block_640 = SHSA(640)
        self.conv1x1 = nn.Conv2d(in_channels=960, out_channels=640, kernel_size=1)

        self.final = nn.Sequential(
            nn.Linear(in_features=640, out_features=960, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=480, out_features=640, bias=True)
        )
    def forward(self, x):
        x = self.shsa_block_960(x)
        # print(x.shape)
        x = self.conv1x1(x)
        x = self.shsa_block_640(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.final(x)
        return x

if __name__ == '__main__':
    model = personal_shsa()
    # 测试模型，传入示例输入数据
    input_tensor = torch.randn(32, 960, 7, 7)  # 示例输入，假设为224x224的RGB图像
    output = model(input_tensor)
    print(output.shape)
