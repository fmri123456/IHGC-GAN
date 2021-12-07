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

# print("开始导入数据...")
# 数据文件的相对地址
DataFile = 'Dual.mat'
data = sio.loadmat(DataFile)
# 导入数据
true = data['feat_LMCI']  # 131*164*70
false = data['feat_EMCI']  # 131*164*70

# 转为张量
True_feat = torch.FloatTensor(true)
False_feat = torch.FloatTensor(false)
# 其他参数
True_dis = generator_conv.set_dis(True_feat)
True_H = generator_conv.set_H(True_dis)
False_dis = generator_conv.set_dis(False_feat)
False_H = generator_conv.set_H(False_dis)
False_weight = generator_conv.set_weight(False_feat)
# 初始化参数
batch_size = 64
num_epoch = 50
# 构建数据集
dataset = TensorDataset(True_feat, True_H, False_feat, False_dis, False_H, False_weight)
# 加载数据集
dataload = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print("数据导入成功！")
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

# 判别器的训练由两部分组成，第一部分是真的图像判别为真，第二部分是假的图片判别为假，在这两个过程中，生成器的参数不参与更新。
# 二进制交叉熵损失和优化器
criterion = nn.CrossEntropyLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

# 开始训练
print("开始训练...")
for epoch in range(num_epoch):
    D_loss = []
    G_loss = []
    for i, (true_feat, true_H, false_feat, false_dis, false_H, false_weight) in enumerate(dataload):
        print('第{}次训练第{}批数据'.format(epoch+1, i+1))
        num_img = true_feat.size(0)
        # ========================================================================训练判别器
        real_img = Variable(true_feat)
        label = np.concatenate((np.zeros(num_img), np.ones(num_img)))
        label = utils.onehot_encode(label)
        label = torch.LongTensor(np.where(label)[1])
        fake_label = label[0:num_img]  # 定义假label为0
        real_label = label[num_img:2*num_img]  # 定义真实label为1
        # 计算 real_img、real_entropy 的损失
        real_out = D(real_img)  # 将真实的图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        # real_scores = real_out  # 越接近1越好
        # real_scores_entropy = real_out_entropy  # 越接近1越好

        # 计算 fake_img的损失
        z = Variable(false_feat)
        fake_G, _, _ ,_,= G(z, false_dis, false_weight)
        fake_out = D(fake_G)  # 判别器判断假的图片
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        # fake_scores = fake_out  # 越接近0越好
        # fake_scores_entropy = fake_out_entropy  # 越接近0越好

        # 反向传播和优化
        d_loss = d_loss_real + d_loss_fake  # 将真假图片的loss加起来
        d_optimizer.zero_grad()  # 每次梯度归零
        d_loss.backward()  # 反向传播
        d_optimizer.step()  # 更新参数

        # =====================================================================训练生成器

        # 计算fake_img损失
        z = Variable(false_feat)
        print("训练生成器G")
        fake_G, fake_W, fake_H,_ = G(z, false_dis, false_weight)  # 生成假的图片
        output = D(fake_G)  # 经过判别器1得到结果
        g_loss = criterion(output, real_label)  #得到假的图片与真实图片label的loss
        # 反向传播和优化
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # 保存每批数据的损失
        D_loss.append(d_loss.item())
        G_loss.append(g_loss.item())

    print('Epoch [{}/{}], d1_loss: {:.6f}, g_loss: {:.6f}'.format(
        epoch+1, num_epoch, sum(D_loss)/len(D_loss), sum(G_loss)/len(G_loss)))


torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator .pth')

