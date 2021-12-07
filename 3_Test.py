import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import scipy.io as sio
import generator_conv
import utils
# 判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(164*70, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 2))

    def forward(self, x):
        # 全连接层
        x = torch.flatten(x, 1, -1)  # 平铺 start_dim=1,end_dim=-1
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x


# 生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.weight = Parameter(torch.FloatTensor(70, 70), requires_grad=True)
        self.weight1 = Parameter(torch.FloatTensor(70, 70), requires_grad=True)
        self.reset_parameters()
        # self.linear = nn.Sequential(nn.Linear(70, 70))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, X, dis, weight):
        delta_W =[]
        G, W, H,dis1,delta_W1 = generator_conv.conv1(X, dis, weight)
        G = torch.matmul(G, self.weight)
        # G = self.linear(G)
        G = F.leaky_relu(G)
        delta_W.append(delta_W1)
        G , W , H,dis1,delta_W1 = generator_conv.conv1(G, dis1, W)
        G = torch.matmul(G , self.weight1)
        # G = self.linear(G)
        G = F.leaky_relu(G)
        delta_W.append(delta_W1)
        # print("生成器卷积训练完成")
        return G, W, H,delta_W

D = discriminator()
G = generator()
D.load_state_dict(torch.load('discriminator .pth'))
G.load_state_dict(torch.load('generator.pth'))
DataFile = 'Dual.mat'
data = sio.loadmat(DataFile)
feat_66 = data['feat_test']
EMCI = feat_66[:33]  # 33*164*70
LMCI = feat_66[33:]  # 33*164*70
feat_66 = torch.FloatTensor(feat_66)
EMCI = torch.FloatTensor(EMCI)
LMCI = torch.FloatTensor(LMCI)
# -------------------------------------------------------------------------------
# print("模型训练完成，开始风险预测...")
feat_dis = generator_conv.set_dis(feat_66)
EMCI_dis = generator_conv.set_dis(EMCI)
LMCI_dis = generator_conv.set_dis(LMCI)
# 对33个距离矩阵求平均
EMCI_dis_mean = EMCI_dis[0:33].mean(dim=0).unsqueeze(0)  # 保持1*164*164的维度
LMCI_dis_mean = LMCI_dis[0:33].mean(dim=0).unsqueeze(0)

# 用平均距离矩阵构建超图
EMCI_H_mean = generator_conv.set_H(EMCI_dis_mean)
LMCI_H_mean = generator_conv.set_H(LMCI_dis_mean)
feat_W = generator_conv.set_weight(feat_66)
EMCI_W = generator_conv.set_weight(EMCI)

generator_G,_,_,_ = G(feat_66, feat_dis, feat_W)

generator_dis = generator_conv.set_dis(generator_G)
generator_H = generator_conv.set_H(generator_dis)
true_pred = utils.pred(LMCI_H_mean, generator_H).reshape(66, 1)
false_pred = utils.pred(EMCI_H_mean, generator_H).reshape(66, 1)


# 将两个矩阵按行拼接:hstack
pred = np.hstack((true_pred, false_pred))  # shape:66*2
pred = torch.FloatTensor(pred)
# 定义测试数据标签
label_pred = np.concatenate((np.ones(33), np.zeros(33)))
label_pred = utils.onehot_encode(label_pred)
label_pred = torch.LongTensor(np.where(label_pred)[1])
# 风险预测准确率

similarity = np.zeros((66, 2))
for k in range(66):
    if pred[k,0]> pred[k,1]:
        similarity[k,0] = pred[k,0]
        similarity[k,1] = 1-pred[k,0]
    else:
        similarity[k ,1] = pred[k , 1]
        similarity[k , 0] = 1 - pred[k , 1]

Acc_pred = utils.accuracy(torch.Tensor(similarity), label_pred)
print('Acc_pred: {:.6f}'.format(Acc_pred))




import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
#画ROC曲线
y_test = label_pred.numpy()
similarity = torch.FloatTensor(similarity)
y_score = F.softmax(similarity,1)
y_score = y_score.detach().numpy()
y_scores = y_score[0:66,1]

fpr,tpr,thr = roc_curve(y_test,y_scores)

roc_auc = auc(fpr,tpr)

lw = 2
plt.figure(figsize=(10 , 10))
plt.plot(fpr , tpr , color='darkorange' ,
         lw=lw , label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0 , 1] , [0 , 1] , color='navy' , lw=lw , linestyle='--')
plt.xlim([0.0 , 1.0])
plt.ylim([0.0 , 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

def stastic_indicators(output,labels):
    TP = ((output.max(1)[1] == 1) & (labels == 1)).sum()
    TN = ((output.max(1)[1] == 0) & (labels == 0)).sum()
    FN = ((output.max(1)[1] == 0) & (labels == 1)).sum()
    FP = ((output.max(1)[1] == 1) & (labels == 0)).sum()
    return TP,TN,FN,FP
TP , TN , FN , FP = stastic_indicators(similarity, y_test)
ACC = (TP + TN) / (TP + TN + FP + FN)
SEN = TP / (TP + FN)
SPE = TN /(FP + TN)
BAC = (SEN + SPE) / 2


np.save('TPR.npy',tpr)
np.save('FPR.npy',fpr)
np.save('ACC.npy',ACC)
np.save('SEN.npy',SEN)
np.save('SPE.npy',SPE)
np.save('BAC.npy',BAC)


