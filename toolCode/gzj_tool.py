from itertools import combinations

import numpy as np
import torch
from sklearn.model_selection import KFold
from scipy import interpolate


def calculate_roc(thresholds, embeddings, embeddings1_idx, embeddings2_idx, actual_issame, nrof_folds=10, pca=0):
    nrof_pairs = min(len(actual_issame), embeddings1_idx.shape[0])
    # 阈值列表的长度
    nrof_thresholds = len(thresholds)
    # 使用KFold 创建一个交叉验证迭代器,将数据分为nrof_folds个折叠
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    # 初始化一些数组
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    dist = get_dist(embeddings, embeddings1_idx, embeddings2_idx)  # 特征差异

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros(nrof_thresholds)

        for threshold_idx, threshold in enumerate(thresholds):  # 对于train_Set
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):  # 对于test_Set
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds  # TAR\FAR\ACC

def get_dist(embeddings, embed1_index, embed2_index):
    """Compute distance between pairs of embeddings"""
    dist = []
    for ed1_idx, ed2_idx in zip(embed1_index, embed2_index):
        ed1 = embeddings[int(ed1_idx)]
        ed2 = embeddings[int(ed2_idx)]
        # Ensure embeddings are torch tensors
        ed1 = torch.tensor(ed1) if not isinstance(ed1, torch.Tensor) else ed1
        ed2 = torch.tensor(ed2) if not isinstance(ed2, torch.Tensor) else ed2
        dist.append(cos_sim(ed1, ed2))
    return torch.tensor(dist)

def cos_sim(a, b):
    """Cosine similarity between vector a and b"""
    a = torch.tensor(a) if not isinstance(a, torch.Tensor) else a
    b = torch.tensor(b) if not isinstance(b, torch.Tensor) else b
    a, b = a.reshape(-1), b.reshape(-1)
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)  # 若dist(相似度) > threshold(足够的理由认为是可以相信的),则返回True
    tp = torch.sum(np.logical_and(predict_issame, actual_issame))
    fp = torch.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = torch.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = torch.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.shape[0]
    return tpr, fpr, acc

def get_issame(total_class_num, single_class_num):
    """
    两个参数: 总的测试集的类别数目, 单个类中图片的个数
    返回值: 匹配的方式(下标索引), 每个匹配对是否为同一类的issame数组
    """
    total_pic_num = total_class_num * single_class_num
    a = [i for i in range(total_pic_num)]
    num_elements_per_combination = 2
    val_pair = list(combinations(a, num_elements_per_combination))
    issame = []
    for i in range(1, total_class_num + 1):
        for j in range(1, single_class_num + 1):
            issame.extend([1] * (single_class_num - j))
            issame.extend([0] * (total_class_num - i) * single_class_num)
    return val_pair, issame

def evaluate(val_pair, actual_issame, embeddings, nrof_folds=10, pca=0):
    # nrof_folds=8
    # Calculate evaluation metrics
    thresholds = np.arange(0.1, 1.0, 0.001)  # 在Fv识别中的threshold用的余弦相似度,故会改范围
    # embeddings1_idx = embeddings[0::2]      # 以步长为2提取
    # embeddings2_idx = embeddings[1::2]
    embeddings1_idx = []
    embeddings2_idx = []
    for (i, j) in val_pair:
        embeddings1_idx.append(i)
        embeddings2_idx.append(j)
    embeddings1_idx = torch.Tensor(embeddings1_idx)
    embeddings2_idx = torch.Tensor(embeddings2_idx)
    # 计算 tar/far/acc
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings, embeddings1_idx, embeddings2_idx,
                                                        np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    # 计算eer
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    # 计算TAR@FAR=0.01
    TAR_when_fixed_FAR_target, _, _ = calculate_tar(thresholds, embeddings,
                                                    embeddings1_idx, embeddings2_idx, torch.Tensor(actual_issame), 0.01)
    return tpr, fpr, accuracy, best_thresholds, eer, eer_threshold, TAR_when_fixed_FAR_target

def compute_eer(far, tar, threshold):
    """
    Compute EER for FAR and TAR.
    EER：当FRR == FAR 时的值
    """
    values = np.abs(1 - tar - far)  # 1 - TAR = FRR, 所以本行表达式就是在说 |FRR - FAR| -->绝对值
    idx = np.argmin(values)         # 求出最小的那个(即FRR-FAR的绝对值越靠近零)
    eer = far[idx]                  # 此时FAR or FRR的值就是EER的值
    thr = threshold[idx]            # 返回此时的thr
    return eer, thr


def calculate_tar(thresholds, embeddings, embeddings1_idx, embeddings2_idx, actual_issame, far_target, nrof_folds=10):
    nrof_pairs = min(len(actual_issame), embeddings1_idx.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tar = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    dist = get_dist(embeddings, embeddings1_idx, embeddings2_idx)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_tar_far(threshold, dist[train_set], actual_issame[train_set])

        far_train += np.random.uniform(0, 1e-6, far_train.shape)
        # print(far_train)

        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear', fill_value='extrapolate')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_idx], far[fold_idx] = calculate_tar_far(threshold, dist[test_set], actual_issame[test_set])

    tar_mean = np.mean(tar)
    far_mean = np.mean(far)
    tar_std = np.std(tar)
    return tar_mean, tar_std, far_mean

def calculate_tar_far(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    true_accept = torch.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = torch.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = torch.sum(actual_issame)
    n_diff = torch.sum(np.logical_not(actual_issame))
    tar = float(true_accept) / float(n_same)    # TAR
    far = float(false_accept) / float(n_diff)   # FAR
    return tar, far