import numpy as np
from pyexpat import features
# from vit_pytorch import ViT
import models
import torch
from torch.utils.data import Dataset, DataLoader
from toolCode.accuracy import accuracy
from toolCode.averageMeter import AverageMeter
from toolCode.gzj_tool import *
from toolCode.lossNet import CosFaceLoss
from toolCode.dic import DictDataset
from test_modify_models.center_net import CenterLossNet


class Client(object):

    def __init__(self, conf, model, train_dataset, eval_dataset, train_classes, eval_classes, per_num, file_path, warm=1, id=-1):

        self.conf = conf
        self.model_v = models.get_model().cuda()
        self.model_shu = models.get_model().cuda()
        self.shared_model = models.get_model().cuda()
        self.local_model = models.client_local_model().cuda()
        self.client_id = id

        self.train_dataset = train_dataset

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], shuffle=True,
                                                        num_workers=16, pin_memory=True)

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=conf["batch_size"], shuffle=False,
                                                       num_workers=16, pin_memory=True)

        self.num_samples = len(train_dataset)
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.per_num = per_num
        self.file_path = file_path

        self.warm = warm

        self.center_net = CenterLossNet(train_classes, 640).cuda()

        self.learning_rate = 0.01

        # 定义优化器，包含 shared_model, local_model 和 center_net 的参数
        self.optimizer_1 = torch.optim.SGD(
            params=[
                {'params': self.shared_model.parameters()},
                {'params': self.center_net.parameters()},
                {'params': self.local_model.parameters()}
            ],
            lr=self.learning_rate,  # 学习率
            weight_decay=0.0001,  # 常用的动量值
            momentum=0.0001
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_1, step_size=10, gamma=0.1)


        # self.optimizer_1 = torch.optim.Adam(
        #     params=[
        #         {'params': self.shared_model.parameters()},
        #         {'params': self.center_net.parameters()},
        #         {'params': self.local_model.parameters()}
        #     ],
        #     lr=self.learning_rate,  # 学习率，与 SGD 的设置方式一致
        #     betas=(0.9, 0.999),  # 默认值，适合大多数场景
        #     eps=1e-8,  # 防止数值问题的 epsilon
        #     weight_decay=0  # 如果需要 L2 正则化，可设置为非零值
        # )

        # self.prototype = {}

        print("Finish Creating Client: ", self.client_id)
        # print(self.shared_model)
        # print(self.local_model)

    def local_train(self, model,client_losses,idx):

        for name, param in model.state_dict().items():
            self.shared_model.state_dict()[name].copy_(param.clone())
        # if self.warm == 0:
        #     self.learning_rate = self.learning_rate * 0.98
        #     for param_group in self.optimizer_1.param_groups:
        #         param_group['lr'] = self.learning_rate
        if self.warm >= 10:
                self.learning_rate = 0.1
                for param_group in self.optimizer_1.param_groups:
                    param_group['lr'] = self.learning_rate
        # else:
        #     self.learning_rate = self.learning_rate * 0.98

            # 打印当前学习率
        print(f"Current LR: {self.learning_rate}")
        self.shared_model.train()
        self.local_model.train()

        cos_face = CosFaceLoss(in_features=640, out_features=self.train_classes).cuda()  # 初始化 CosFaceLos
        # cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()  # 实例化损失函数
        loss_com=0

        for e in range(self.conf["local_epochs"]):

            losses = AverageMeter()

            # 一次训练
            for i, (input, target) in enumerate(self.train_loader):
                self.optimizer_1.zero_grad()

                input = input.cuda()
                target = target.cuda()

                features = self.shared_model(input)  # 提取线性层之前的特征

                # for index in range(target.size(0)):  # 遍历所有数据
                #     key = target[index].item()  # 获取类别值
                #     if key in self.prototype:
                #         self.prototype[key] += features[index]  # 如果存在类别，累加特征向量
                #     else:
                #         self.prototype[key] = features[index]  # 如果不存在，直接赋值

                output = self.local_model(features)

                # 使用 CosFaceLoss 计算损失
                loss_cf = cos_face(output, target)

                # loss_ce = cross_entropy_loss(output, target)
                center_loss = self.center_net(output, target)
                # loss = 0.4 * loss_ce + 0.005 * center_loss  # 合并损失项
                # print(loss_cf)
                loss = loss_cf + center_loss * 0
                # loss = loss_ce
                loss.backward()

                losses.update(loss.item(), input.size(0))

                self.optimizer_1.step()

            # for key in self.prototype:
            #     self.prototype[key] = self.prototype[key].detach() / self.per_num  # 逐元素除以 divisor
            loss_com +=losses.val
            print("client-{}---Epoch {} done -> \tLoss {loss.val:.3f}".format(self.client_id, e+1, loss=losses))
            log_message = "client-{}---Epoch {} done -> \tLoss {loss.val:.3f}\n".format(
                self.client_id, e + 1, loss=losses
            )

            # 写入到文件
            with open(self.file_path, "a") as file:
                file.write(log_message)
        self.warm+=1
        client_losses[idx].append(loss_com/5)
        print(client_losses)
        print("client-{} all done".format(self.client_id))

        diff = dict()
        original_values = dict()

        for name, data in self.shared_model.state_dict().items():
            # Save the original values first
            original_values[name] = data.cuda()
            # Calculate the difference
            diff[name] = (data - model.state_dict()[name].cuda())

        # Return both diff and original_values dictionaries
        return diff
    # def local_train_vit(self, model):
    #
    #     self.model_v.load_state_dict(model.state_dict())  # 更安全的方式
    #
    #
    #     # 3. 定义优化器（只在这个方法内部使用）
    #     optimizer_vit = torch.optim.Adam(
    #         self.model_v.parameters(),
    #         lr=self.learning_rate,
    #
    #         weight_decay=5e-4
    #     )
    #
    #     optimizer_center = torch.optim.SGD(
    #         self.center_net.parameters(),
    #         lr=0.05
    #     )
    #
    #
    #     # if self.warm == 0:
    #     #     self.learning_rate = self.learning_rate * 0.98
    #     #     for param_group in self.optimizer_1.param_groups:
    #     #         param_group['lr'] = self.learning_rate
    #     #
    #     #     # 打印当前学习率
    #     #     print(f"Current LR: {self.learning_rate}")
    #
    #     self.model_v.train()
    #     self.local_model.train()
    #     cos_face = CosFaceLoss(in_features=640, out_features=self.train_classes).cuda()  # 初始化 CosFaceLos
    #     # cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()  # 实例化损失函数
    #     for e in range(self.conf["local_epochs"]):
    #
    #         losses = AverageMeter()
    #
    #         # 一次训练
    #         for i, (input, target) in enumerate(self.train_loader):
    #             optimizer_vit.zero_grad()
    #             optimizer_center.zero_grad()
    #             input = input.cuda()
    #             target = target.cuda()
    #             features = self.model_v(input)
    #
    #             output = self.local_model(features)
    #             # 使用 CosFaceLoss 计算损失
    #             loss_cf = cos_face(output, target)
    #             # loss_ce = cross_entropy_loss(output, target)
    #             center_loss = self.center_net(output, target)
    #             # loss = 0.4 * loss_ce + 0.005 * center_loss  # 合并损失项
    #             # print(loss_cf)
    #             loss = loss_cf + center_loss * 0.003
    #             # loss = loss_ce
    #             loss.backward()
    #
    #             losses.update(loss.item(), input.size(0))
    #
    #             optimizer_vit.step()
    #             optimizer_center.step()
    #
    #         # for key in self.prototype:
    #         #     self.prototype[key] = self.prototype[key].detach() / self.per_num  # 逐元素除以 divisor
    #
    #         print("client-{}---Epoch {} done -> \tLoss {loss.val:.3f}".format(self.client_id, e+1, loss=losses))
    #         log_message = "client-{}---Epoch {} done -> \tLoss {loss.val:.3f}\n".format(
    #             self.client_id, e + 1, loss=losses
    #         )
    #
    #         # 写入到文件
    #         with open(self.file_path, "a") as file:
    #             file.write(log_message)
    #
    #     print("client-{} all done".format(self.client_id))
    #
    #     diff = dict()
    #     original_values = dict()
    #
    #     for name, data in self.model_v.state_dict().items():
    #         # Save the original values first
    #         original_values[name] = data.cuda()
    #         # Calculate the difference
    #         diff[name] = (data.cuda() - model.state_dict()[name].cuda())
    #         # Return both diff and original_values dictionaries
    #     return diff

    def local_train_shu(self, model):

        for name, param in model.state_dict().items():
            self.model_shu.state_dict()[name].copy_(param.clone())
        if self.warm == 0:
            self.learning_rate = self.learning_rate * 0.98
            for param_group in self.optimizer_1.param_groups:
                param_group['lr'] = self.learning_rate
            # 打印当前学习率
            print(f"Current LR: {self.learning_rate}")
        self.model_shu.train()
        self.local_model.train()
        cos_face = CosFaceLoss(in_features=640, out_features=self.train_classes).cuda()  # 初始化 CosFaceLos
        # cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()  # 实例化损失函数
        for e in range(self.conf["local_epochs"]):

            losses = AverageMeter()

            # 一次训练
            for i, (input, target) in enumerate(self.train_loader):
                self.optimizer_1.zero_grad()

                input = input.cuda()
                target = target.cuda()
                features = self.model_shu(input)
                output = self.local_model(features)  # 提取线性层之前的特征

                # for index in range(target.size(0)):  # 遍历所有数据
                #     key = target[index].item()  # 获取类别值
                #     if key in self.prototype:
                #         self.prototype[key] += features[index]  # 如果存在类别，累加特征向量
                #     else:
                #         self.prototype[key] = features[index]  # 如果不存在，直接赋值


                # 使用 CosFaceLoss 计算损失
                loss_cf = cos_face(output, target)

                # loss_ce = cross_entropy_loss(output, target)
                center_loss = self.center_net(output, target)
                # loss = 0.4 * loss_ce + 0.005 * center_loss  # 合并损失项
                # print(loss_cf)
                loss = loss_cf + center_loss * 0.003
                # loss = loss_ce
                loss.backward()

                losses.update(loss.item(), input.size(0))

                self.optimizer_1.step()

            # for key in self.prototype:
            #     self.prototype[key] = self.prototype[key].detach() / self.per_num  # 逐元素除以 divisor

            print("client-{}---Epoch {} done -> \tLoss {loss.val:.3f}".format(self.client_id, e + 1, loss=losses))
            log_message = "client-{}---Epoch {} done -> \tLoss {loss.val:.3f}\n".format(
                self.client_id, e + 1, loss=losses
            )

            # 写入到文件
            with open(self.file_path, "a") as file:
                file.write(log_message)

        print("client-{} all done".format(self.client_id))

        diff = dict()
        original_values = dict()

        for name, data in self.shared_model.state_dict().items():
            # Save the original values first
            original_values[name] = data.cuda()
            # Calculate the difference
            diff[name] = (data - model.state_dict()[name].cuda())

        # Return both diff and original_values dictionaries
        return diff

    def local_test(self):
        with torch.no_grad():
            self.shared_model.eval()
            self.local_model.eval()

            # 收集特征嵌入
            all_embeddings = []

            for i, (input, target) in enumerate(self.eval_loader):
                input = input.cuda()

                features = self.shared_model(input)
                output = self.local_model(features)
                output = output.cpu().numpy()

                all_embeddings.append(output)

            embeddings = np.concatenate(all_embeddings, axis=0)

            val_pair, is_same = get_issame(self.eval_classes, self.per_num)

            tpr, fpr, _accuracy, best_thresholds, eer, eer_threshold, TAR_when_fixed_FAR_target = evaluate(
                val_pair, is_same, embeddings)
        return eer * 100, eer_threshold, TAR_when_fixed_FAR_target * 100

    def local_test2(self):
        with torch.no_grad():
            self.model_v.eval()
            self.local_model.eval()

            # 收集特征嵌入
            all_embeddings = []

            for i, (input, target) in enumerate(self.eval_loader):
                input = input.cuda()

                features = self.shared_model(input)
                output = self.local_model(features)
                output = output.cpu().numpy()

                all_embeddings.append(output)

            embeddings = np.concatenate(all_embeddings, axis=0)

            val_pair, is_same = get_issame(self.eval_classes, self.per_num)

            tpr, fpr, _accuracy, best_thresholds, eer, eer_threshold, TAR_when_fixed_FAR_target = evaluate(
                val_pair, is_same, embeddings)
        return eer * 100, eer_threshold, TAR_when_fixed_FAR_target * 100

    def local_test3(self):
        with torch.no_grad():
            self.model_shu.eval()


            # 收集特征嵌入
            all_embeddings = []

            for i, (input, target) in enumerate(self.eval_loader):
                input = input.cuda()

                output = self.model_shu(input)
                output = output.cpu().numpy()

                all_embeddings.append(output)

            embeddings = np.concatenate(all_embeddings, axis=0)

            val_pair, is_same = get_issame(self.eval_classes, self.per_num)

            tpr, fpr, _accuracy, best_thresholds, eer, eer_threshold, TAR_when_fixed_FAR_target = evaluate(
                val_pair, is_same, embeddings)
        return eer * 100, eer_threshold, TAR_when_fixed_FAR_target * 100


