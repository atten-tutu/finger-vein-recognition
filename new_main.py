import json
import os
import torch
import datasets
from client import *
from server import *
import datetime
import copy

import json
import random
import time
import torch
from server import Server
from client import Client

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import numpy as np
import copy


def search(all_weights, all_choose_k, random_index_k, start_index, now_path, res, server, cal_num):
    if (start_index >= len(random_index_k)):
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        # 计算client的中心_1 求和
        for i in range(len(all_weights)):
            if (len(all_weights[i])) != 0:  # 如果被选中过
                temp = all_weights[i][now_path[i]]
                for name, params in temp.items():
                    weight_accumulator[name] += params
        # 计算client的中心_2 除数量
        for name, params in weight_accumulator.items():
            weight_accumulator[name] = weight_accumulator[name] / cal_num
        # 计算client中心到各client的距离
        temp_res = 0
        for i in range(len(all_weights)):
            if (len(all_weights[i])) != 0:  # 如果被选中过
                temp = all_weights[i][now_path[i]]
                for name, params in temp.items():
                    temp_res += np.linalg.norm((weight_accumulator[name] - params).cpu())  # tensor转numpy
        if (temp_res < res):
            print("出现")
            for i in range(len(now_path)):
                all_choose_k[i] = now_path[i]
                # all_choose_k = copy.copy(now_path) #注意这里不能这样写
            res = temp_res
        return
    for every in range(len(all_weights[random_index_k[start_index]])):
        now_path[random_index_k[start_index]] = every
        search(all_weights, all_choose_k, random_index_k, start_index + 1, now_path, res, server, cal_num)


def main(test_epoch):

    json_file_path = 'conf.json'

    # 读取 JSON 配置文件
    with open(json_file_path, 'r') as f:
        conf = json.load(f)

    torch.manual_seed(42)

    train_hkpu_dir = "../fingerData/hkpu/train"
    test_hkpu_dir = "../fingerData/hkpu/test"
    train_vera_dir = "../fingerData/vera/train"
    test_vera_dir = "../fingerData/vera/test"
    train_utfvp_dir = "../fingerData/utfvp/train"
    test_utfvp_dir = "../fingerData/utfvp/test"
    train_bnu_dir = "../fingerData/bnu/train"
    test_bnu_dir = "../fingerData/bnu/test"
    train_nupt_dir = "../fingerData/nupt/train"
    test_nupt_dir = "../fingerData/nupt/test"
    train_sdu_dir = "../fingerData/sdu/train"
    test_sdu_dir = "../fingerData/sdu/test"

    train_hkpu_datasets, eval_hkpu_datasets = datasets.get_dataset(train_hkpu_dir, test_hkpu_dir)
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
    output_dir = "logs"  # 定义输出目录
    file_path = os.path.join(output_dir, file_name)
    print(file_path)

    # Initialize clients, transferring the model to the correct device
    clients = [
        Client(conf, server.global_model, train_hkpu_datasets, eval_hkpu_datasets, train_classes[0], eval_classes[0], per_eval_num[0], file_path, 1, "hkpu"),
        Client(conf, server.global_model, train_vera_datasets, eval_vera_datasets, train_classes[1], eval_classes[1], per_eval_num[1], file_path, 1, "vera"),
        Client(conf, server.global_model, train_utfvp_datasets, eval_utfvp_datasets, train_classes[2], eval_classes[2], per_eval_num[2], file_path, 1, "utfvp"),
        Client(conf, server.global_model, train_bnu_datasets, eval_bnu_datasets, train_classes[3], eval_classes[3], per_eval_num[3], file_path, 1, "bnu"),
        Client(conf, server.global_model, train_nupt_datasets, eval_nupt_datasets, train_classes[4], eval_classes[4], per_eval_num[4], file_path, 1, "nupt"),
        Client(conf, server.global_model, train_sdu_datasets, eval_sdu_datasets, train_classes[5], eval_classes[5], per_eval_num[5], file_path, 1, "sdu"),
    ]
    # for c in clients:
    #     eer, threshold, tar_far = c.local_test()
    #     print(f"{c.client_id}---EER:{eer:.4f}, Threshold:{threshold:.2f},TAR@FAR=0.01:{tar_far:.4f}\t\n")
    #     log_message = f"{c.client_id}---EER:{eer:.4f}, Threshold:{threshold:.2f},TAR@FAR=0.01:{tar_far:.4f}\t\n"
    #
    #     # 写入到文件
    #     with open(file_path, "a") as file:
    #         file.write(log_message)
    #
    # print("\n")

    start_time = datetime.datetime.now()
    all_weights = [[] for i in range(6)]  # 全局缓存表，缓存的是client的3个绝对参数
    all_choose_k = [0 for i in range(6)]  # 默认选择的是第一个
    memory_k = 3
    #开始训练
    for e in range(test_epoch):
        print(f"总轮次：{e + 1}\n")
        log_message = f"总轮次：{e + 1}\n"
        # 写入到文件
        with open(file_path, "a") as file:
            file.write(log_message)
        print(start_time)

        round_all_weights = [{} for i in range(6)]  # 本轮缓存表，缓存的是相对参数
        weight_accumulator2 = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator2[name] = torch.zeros_like(params).to(device) # 初始化上面的参数字典，大小和全局模型相同

        # weight_accumulator2 = {name: [] for name in server.global_model.state_dict().keys()}
        #每个键对应的值是一个列表，用来收集所有客户端对该参数的更新。

        # 收集客户端样本数，用于后续加权平均。
        num_samples_per_client = []

        for c_index, c in enumerate(clients):
            num_samples = c.num_samples
            num_samples_per_client.append(num_samples)

            diff = c.local_train_vit(server.global_model)#以全局模型 server.global_model 为基础进行本地训练。
            client_temp = copy.deepcopy(diff)  # 复制当前客户端的更新（训练后的参数变化）

            if (len(all_weights[c_index]) >= memory_k):
                del all_weights[c_index][0]  # LRU删除
                all_weights[c_index].append({})  # 添加当前
                for name, data in server.global_model.state_dict().items():
                    # 最后一个应该是memory_k-1
                    all_weights[c_index][memory_k - 1][name] = client_temp[name]
            else:
                all_weights[c_index].append({})  # 添加当前
                for name, data in server.global_model.state_dict().items():
                    all_weights[c_index][len(all_weights[c_index]) - 1][name] = client_temp[name]  # 记录全局缓存表
            round_all_weights[c_index] = diff  # 记录本地缓存表模型变换量，这时候只是给一个初值

            #diff=local_model_params−global_model_params


        # 参数聚合完成全局模型参数的更新。加权平均样本数加权：
        #
        # 若轮次较少，固定分配权重（均等权重）。
        # 若轮次较多，按客户端样本数分配权重。
        # 聚合客户端更新：
        #
        # 按权重累积所有客户端的更新值，生成最终的更新张量。
        # 更新全局模型：
        #
        # 将聚合后的更新值直接加到全局模型参数上。
        if e < 14:
            # 此时warmup阶段，用策略一
            # 这里可以只改被选中的
            for i in range(6):
                if (len(all_weights[i]) != 0):
                    all_choose_k[i] = len(all_weights[i]) - 1
        else:
            # 搜索函数，寻找这个时候的被选中的client应该选中的model,即最后返回的结果在all_choose_k中体现
            now_path = copy.deepcopy(all_choose_k)
            result = 99999999999999999
            random_index_k = list(range(6))  # 确保所有6个客户端都参与
            search(all_weights, all_choose_k,  random_index_k,0, now_path, result, server, 6)
        # random 搜索
        # 遍历所有客户端
        for c_index in range(len(clients)):
            for name, data in server.global_model.state_dict().items():
                round_all_weights[c_index][name] = all_weights[c_index][all_choose_k[c_index]][name] - \
                                                   server.global_model.state_dict()[name]

                # 记录选中的部分
            for name, params in server.global_model.state_dict().items():
                weight_accumulator2[name].add_(round_all_weights[c_index][name])

                # 参数聚合完成全局模型参数的更新。加权平均
        server.model_aggregate(weight_accumulator2, num_samples_per_client, e)

        end_time = datetime.datetime.now()
        print(end_time - start_time)

        # 热身阶段完毕
        if (e+1) == 15:
            print("热身阶段结束")
            for c in clients:
                c.warm = 0
                print(f"id：{c.client_id} warm:{c.warm}\n")

        if (e+1) == test_epoch:
            for c in clients:
                eer, threshold, tar_far = c.local_test3()
                print(f"{c.client_id}---EER:{eer:.4f}, Threshold:{threshold:.2f},TAR@FAR=0.01:{tar_far:.4f}\t\n")
                log_message = f"{c.client_id}---EER:{eer:.4f}, Threshold:{threshold:.2f},TAR@FAR=0.01:{tar_far:.4f}\t\n"

                # 写入到文件
                with open(file_path, "a") as file:
                    file.write(log_message)


if __name__ == '__main__':
    main(20)
