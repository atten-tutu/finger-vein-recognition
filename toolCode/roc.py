import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d
# 1.载入.npy文件;pred_images.npy文件是模型预测的结果的汇总；gt_images.npy是ground truth的汇总
pred = np.load('pred_images.npy')
gt = np.load('gt_images.npy')
# 2.定义一个画布
plt.figure(1)
# 3.计算fpr、tpr及roc曲线的面积
fpr, tpr, thresholds = roc_curve((gt), pred)
roc_auc = auc(fpr, tpr)
# 4.绘制roc曲线
plt.plot(fpr, tpr, label='UNet (area = {:.4f})'.format(roc_auc), color='blue')
# 5.格式个性化
font1 = {
'weight' : 'normal',
'size'   : 14, }
plt.xlabel("FPR (False Positive Rate)", font1)
plt.ylabel("TPR (True Positive Rate)", font1)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.axis([0, 1, 0.70, 1])
plt.title('ROC Curve', font1)
plt.show()
print('Done!')
