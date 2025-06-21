import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch              # 导入PyTorch库
import torchvision.transforms as transforms  # 导入图像预处理模块
import timm               # 导入timm库以获取EfficientNetV2模型
from PIL import Image     # 导入Pillow，用于图像处理
import datasets
from client import *
from server import *
import datetime

def main(test_epoch):

    json_file_path = 'conf.json'

    # 读取 JSON 配置文件
    with open(json_file_path, 'r') as f:
        conf = json.load(f)

    torch.manual_seed(42)

    train_hkpu_dir = "fingerData/hkpu/train"
    test_hkpu_dir = "fingerData/hkpu/test"
    train_vera_dir = "fingerData/vera/train"
    test_vera_dir = "fingerData/vera/test"
    train_utfvp_dir = "fingerData/utfvp/train"
    test_utfvp_dir = "fingerData/utfvp/test"
    train_bnu_dir = "fingerData/bnu/train"
    test_bnu_dir = "fingerData/bnu/test"
    train_nupt_dir = "fingerData/nupt/train"
    test_nupt_dir = "fingerData/nupt/test"
    train_sdu_dir = "fingerData/sdu/train"
    test_sdu_dir = "fingerData/sdu/test"

    train_hkpu_datasets, eval_hkpu_datasets = datasets.get_dataset(train_hkpu_dir,test_hkpu_dir)
    train_vera_datasets, eval_vera_datasets = datasets.get_dataset(train_vera_dir, test_vera_dir)
    train_utfvp_datasets, eval_utfvp_datasets = datasets.get_dataset(train_utfvp_dir, test_utfvp_dir)
    train_bnu_datasets, eval_bnu_datasets = datasets.get_dataset(train_bnu_dir, test_bnu_dir)
    train_nupt_datasets, eval_nupt_datasets = datasets.get_dataset(train_nupt_dir, test_nupt_dir)
    train_sdu_datasets, eval_sdu_datasets = datasets.get_dataset(train_sdu_dir, test_sdu_dir)

    server = Server(conf)
    print("Finish Creating Server")

    # Check if CUDA is available and select the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the global model to the appropriate device (CUDA or CPU)
    server.global_model = server.global_model.to(device)

    # hkpu, vera, utfvp, bnu, nupt, sdu
    train_classes = [250, 176, 288, 480, 1344, 508]
    eval_classes = [62, 44, 72, 120, 336, 128]
    per_eval_num = [6, 2, 4, 10, 10, 6]

    # 获取当前日期和时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 定义文件名
    file_name = f"{current_time}.txt"
    # 确保目录存在（可选）
    output_dir = "logs"  # 定义输出目录、
    file_path = os.path.join(output_dir, file_name)
    print(file_path)

    clients = [
        Client(conf, server.global_model.cuda(), train_hkpu_datasets, eval_hkpu_datasets, train_classes[0], eval_classes[0], per_eval_num[0], file_path,  1, "hkpu"),
        Client(conf, server.global_model.cuda(), train_vera_datasets, eval_vera_datasets, train_classes[1], eval_classes[1], per_eval_num[1], file_path, 1, "vera"),
        Client(conf, server.global_model.cuda(), train_utfvp_datasets, eval_utfvp_datasets, train_classes[2], eval_classes[2], per_eval_num[2], file_path, 1, "utfvp"),
        Client(conf, server.global_model.cuda(), train_bnu_datasets, eval_bnu_datasets, train_classes[3], eval_classes[3], per_eval_num[3], file_path, 1, "bnu"),
        Client(conf, server.global_model.cuda(), train_nupt_datasets, eval_nupt_datasets, train_classes[4], eval_classes[4], per_eval_num[4], file_path, 1, "nupt"),
        Client(conf, server.global_model.cuda(), train_sdu_datasets, eval_sdu_datasets, train_classes[5], eval_classes[5], per_eval_num[5], file_path, 1, "sdu"),
    ]

    print("\n")

    start_time = datetime.datetime.now()
    client_losses = [[],[],[],[],[],[]]
    num_samples_per_client1=[]
    for c in clients:
        num_samples = c.num_samples
        num_samples_per_client1.append(num_samples)
    total_nums = sum(num_samples_per_client1)
    weight = []  # 先创建空列表
    for i in range(len(num_samples_per_client1)):
        weight.append(num_samples_per_client1[i] / total_nums)  # 用 append 添加元素
    print(weight)

    for e in range(test_epoch):
        print("总轮次：{}\n".format(e + 1))
        log_message = "总轮次：{}\n".format(e + 1)
        # 写入到文件
        with open(file_path, "a") as file:
            file.write(log_message)
        print(start_time)
        # candidates = random.sample(clients, conf["k"])

        weight_accumulator = {name: [] for name in server.global_model.state_dict().keys()}

        # 收集客户端样本数
        num_samples_per_client = []

        for idx, c in enumerate(clients):
            num_samples = c.num_samples
            num_samples_per_client.append(num_samples)
            diff = c.local_train(server.global_model,client_losses,idx)
            for name in weight_accumulator:
                weight_accumulator[name].append(diff[name])

        # 参数聚合
        server.model_aggregate(weight_accumulator, num_samples_per_client, e)

        end_time = datetime.datetime.now()
        print(end_time - start_time)

        # # 热身阶段完毕
        # if (e+1) == 15:
        #     print("热身阶段结束")
        #     for c in clients:
        #         c.warm = 0
        #         print("id：{} warm:{}\n".format(c.client_id, c.warm))

        if (e+1) == test_epoch:

            for c in clients:
                eer, threshold, tar_far = c.local_test()
                print("{}---EER:{:.4f}, Threshold:{:.2f},TAR@FAR=0.01:{:.4f}\t\n".format(c.client_id, eer, threshold, tar_far))
                log_message = "{}---EER:{:.4f}, Threshold:{:.2f},TAR@FAR=0.01:{:.4f}\t\n".format(c.client_id, eer, threshold, tar_far)

                # 写入到文件
                with open(file_path, "a") as file:
                    file.write(log_message)
        # 加载之前的 Loss
    history_file = "global_loss.npy"
    if os.path.exists("global_loss.npy"):

        history = np.load(history_file)  # 读取历史数据
    else:
        history = np.empty((0, test_epoch))  # 初始化一个空的二维数组

    global_loss = np.zeros(test_epoch)
    for client_id in range(6):  # 遍历 6 个客户端
        global_loss += np.array(client_losses[client_id]) * weight[client_id]
    history = np.vstack([history, global_loss])

    np.save("global_loss.npy", history)  # 保存为 NumPy 二进制文件

    # # 画图
    # plt.figure(figsize=(8, 5))
    # for i, loss_curve in enumerate(history):
    #     plt.plot(range(test_epoch), loss_curve, linestyle='-', marker='o', label=f'Training {i + 1}')
    #
    # plt.xlabel("Test Epochs")
    # plt.ylabel("Weighted Loss")
    # plt.title("Federated Learning - Global Loss Curve (All Trainings)")
    # plt.legend()
    # plt.grid(True)
    #
    # # 保存图像
    # plt.savefig("global_loss_all_trainings.png", dpi=300)
    # plt.show()

if __name__ == '__main__':
    main(20)