from torch import nn

import models
import torch.utils.data

from toolCode.gzj_tool import *


class Server(object):

    def __init__(self, conf):

        self.conf = conf

        self.global_model = models.get_model().cuda()

    # 模型聚合
    def model_aggregate(self, weight_accumulator, num_samples_per_client, e):
        total_samples = sum(num_samples_per_client)
        for name, data in self.global_model.state_dict().items():
            # 先检查 weight_accumulator 和 num_samples_per_client 是否匹配
            if name not in weight_accumulator or len(weight_accumulator[name]) == 0:
                print(f"Skipping {name} due to empty weight_accumulator.")
                continue

            if len(weight_accumulator[name]) != len(num_samples_per_client):
                print(
                    f"Mismatch: {name} has {len(weight_accumulator[name])} updates, but {len(num_samples_per_client)} clients.")
                continue

            # 确保 update_per_layer 是浮点张量，且在 CUDA 上
            update_per_layer = torch.zeros_like(data, dtype=torch.float32).cuda()

            for client_index, client_data in enumerate(weight_accumulator[name]):
                if client_index >= len(num_samples_per_client):  # 预防索引越界
                    print(f"Skipping client_index {client_index} for {name}, out of range.")
                    continue

                if e + 1 >=10:
                    weight = num_samples_per_client[client_index] / total_samples
                else:
                    weight = 0.1666667

                update_per_layer += client_data * weight

            # 直接加到 data 上，确保数据类型匹配
            data.add_(update_per_layer.to(data.dtype))


