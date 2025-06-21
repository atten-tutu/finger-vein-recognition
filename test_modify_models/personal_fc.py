import torch
import torch.nn as nn

class personal_fc(nn.Module):
    def __init__(self):
        super(personal_fc, self).__init__()
        self.final = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=640, bias=True)
        )

    def forward(self, x):
        x = self.final(x)
        return x

if __name__ == '__main__':
    model = personal_fc()
    # 测试模型，传入示例输入数据
    input_tensor = torch.randn(32, 960)  # 示例输入，假设为224x224的RGB图像
    output = model(input_tensor)
    print(output.shape)
