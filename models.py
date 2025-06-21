import timm
# from vit_pytorch import ViT


import torch
from torchvision import models
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, DenseNet121_Weights
from shufflenet import ShuffleNetV2
from test_modify_models.ghostnet import ghostnet
from test_modify_models.mamba_attention import MobileNetV3_MamBa
from test_modify_models.only_ca import only_ca
from test_modify_models.only_shsa import only_shsa
from test_modify_models.self_attention import MobileNetV3_Attention
from test_modify_models.personal_attention import personal_attention
from test_modify_models.SHSA_NET import SHSA_NET
from test_modify_models.personal_SHSA import personal_shsa
from test_modify_models.personal_fc import personal_fc


def client_local_model():
    # model = personal_attention()
    # model = personal_shsa()
    model = personal_fc()
    return model


# def get_model():
#     # model = MobileNetV3_Attention()
#     # model = MobileNetV3_MamBa()
#     # model = SHSA_NET()
#     #
#     # # return model
#     model = ViT(
#         image_size=224,  # 输入图像大小
#         patch_size=32,  # Patch 大小
#         num_classes=960,  # 目标分类数
#         dim=1024,  # Token 维度
#         depth=6,  # Transformer 层数
#         heads=16,  # 多头注意力
#         mlp_dim=2048,  # MLP 维度
#         dropout=0.1,
#         emb_dropout=0.1
#     )
#     # model = ShuffleNetV2()
#     # model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
#     return model
def get_model(model_name="only_ca"):
    """
    根据传入的模型名称返回不同的模型
    可选模型：ViT, MobileNetV3, GhostNet, ShuffleNetV2

    :param model_name: 模型名称（字符串类型）
    :return: 选定的模型
    """

    # if model_name == "ViT":
    #     model = create_model(
    #         "vit_base_patch16_224",  # 使用 timm 库创建 ViT 模型
    #         num_classes=960,  # 目标分类数
    #         img_size=224  # 输入图像大小
    #     )

    # elif model_name == "MobileNetV3":
    #     # 使用预训练的 MobileNetV3 大模型
    #     model = models.mobilenet_v3_large(pretrained=True)
    #     # 修改分类数为你自己的目标数
    #     model.classifier[3] = nn.Linear(in_features=1280, out_features=960)

    # elif model_name == "GhostNet":
    #     model = create_model(
    #         "ghostnet_1_3",  # GhostNet 版本
    #         num_classes=960  # 目标分类数
    #     )

    # if model_name == "GhostNet":s
    #     # model = ghostnet(num_classes=960, width=1.0, dropout=0.2)
    #     # 修改分类数
    # # 加载预训练的EfficientNetV2模型
    if model_name == "eff":
        model = models.efficientnet_b0(pretrained=True)
        # 修改全连接层
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=960, bias=True)


    elif model_name == "mo3":
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=960)

    elif model_name == "shsa":
        model = SHSA_NET()
    elif model_name == "shu":
        model = models.shufflenet_v2_x1_5(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024, out_features=960, bias=True)
    elif model_name == "res":
        # 获取最后全连接层的输入特征数
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features

        # 将最后的全连接层替换为输出维度为960的新层
        model.fc = nn.Linear(num_features, 960)


    elif model_name == "gho":
        model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        print(model)
    elif model_name == "squ":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1000, kernel_size=(1, 1)),  # 先保留默认的 1000 维输出
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # 变成 (batch, 1000, 1, 1)
            nn.Flatten(),  # 变成 (batch, 1000)
            nn.Linear(1000, 960)  # 线性层将 1000 维转换为 960 维
        )
        print(model)
    elif model_name == "reg":
        model = models.regnet_y_400mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 960)
        # model.fc = torch.nn.Linear(in_features=32, out_features=960, bias=True)
        print(model)
    elif model_name == "dens":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 960)
        # print(model.classifier.weight[:5])  # 打印前 5 行权重
    elif model_name == "inc":
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 960)
        print(model)
        # print(model.classifier.weight[:5])  # 打印前 5 行权重
    elif model_name == "mnas":
        model = models.mnasnet1_0(pretrained=True)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=960, bias=True)
        # model.classifier = nn.Linear(in_features, 960)
        print(model)
        # print(model.classifier.weight[:5])  # 打印前 5 行权重

    # else:
    #     raise ValueError("Unknown model name: {}".format(model_name))
    elif model_name == "only_shsa":
        model = only_shsa()
    elif model_name == "only_ca":
        model = only_ca()

        # print(model.classifier.weight[:5])  # 打印前 5 行权重
    return model


if __name__ == '__main__':
    get_model()