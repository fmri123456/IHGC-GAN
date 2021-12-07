import numpy as np
import torch
# 生成器卷积
def conv1(X,dis,W):
    H = set_H(dis)
    De = set_De(H)
    Dv = set_Dv(H)
    S = set_S(Dv)
    R = set_R(S)
    Ws = set_Ws(W)
    W,delta_W = HyperGraphconv(S, R, W, Ws)
    dis = updata_distant(W, dis)
    H = set_H(dis)
    HT = set_HT(H)
    invDE = set_invDE(De)
    DV2 = set_DV2(Dv)
    G = torch.bmm(DV2, torch.bmm(H, torch.bmm(W, torch.bmm(invDE, torch.bmm(HT, torch.bmm(DV2, X))))))
    return G, W, H,dis,delta_W
# 欧氏距离矩阵dis
def set_dis(X):
    X = X.detach().numpy()
    row = X.shape[0]
    col = X.shape[1]
    # 初始化欧氏距离矩阵
    dis = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                dis[k][i][j] = np.linalg.norm(X[k][i]-X[k][j])
    dis = torch.FloatTensor(dis)
    return dis
# 计算关联矩阵H
def set_H(dis, k=15):
    dis = dis.detach().numpy()
    count = dis.shape[0]
    n_obj = dis.shape[1]
    n_edge = n_obj
    H = np.zeros((count, n_obj, n_edge))
    for i in range(count):
        for center_idx in range(n_obj):
            dis_vec = dis[i, center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            if not np.any(nearest_idx[:k] == center_idx):
                nearest_idx[k - 1] = center_idx
            for node_idx in nearest_idx[:k]:
                H[i, node_idx, center_idx] = 1
    H = torch.FloatTensor(H)
    return H
# 超边度矩阵De
def set_De(H):
    H = H.detach().numpy()
    row = H.shape[0]
    col = H.shape[1]
    De = np.zeros((row, col, col))
    for i in range(row):
        for j in range(col):
            De[i, j, j] = np.sum(H[i, :, j])
    return De
# 节点度矩阵Dv
def set_Dv(H):
    H = H.detach().numpy()
    row = H.shape[0]
    col = H.shape[1]
    Dv = np.zeros((row, col, col))
    for i in range(row):
        for j in range(col):
            Dv[i, j, j] = np.sum(H[i][j])
    return Dv
# 节点影响力信息传播量矩阵S
def set_S(D,alpha=0.8):
    row = D.shape[0]
    col = D.shape[1]
    S = np.zeros((row, col, col))
    for i in range(row):
        for j in range(col):
            S[i, j] = D[i, j, j]
            S[i, j, j] = 0
    S1 = alpha * S
    return S1
# 节点影响力信息接收比例矩阵R
def set_R(S):
    row = S.shape[0]
    col = S.shape[1]
    R = np.zeros((row, col, col))
    for i in range(row):
        for j in range(col):
            S[i, j, j] = 1
    # 元素倒置
    temp = np.reciprocal(S)
    for i in range(row):
        for j in range(col):
            temp[i, j, j] = 0
    # 矩阵转置
    for i in range(row):
        R[i] = temp[i].T
    return R
# 权重矩阵W
def set_weight(X):
    X = X.detach().numpy()
    # 节点个数
    row = X.shape[0]
    # 特征个数
    col = X.shape[1]
    # 初始化权重矩阵
    W = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                if j != i:
                    W[k][i][j] = np.min(np.corrcoef(X[k][i], X[k][j]))
    W = weight_threshold(W)
    W = torch.FloatTensor(W)
    return W
# 权重矩阵阈值化(关联系数最小的10%元素置0)
def weight_threshold(W):
    row = W.shape[0]
    col = W.shape[1]
    result = np.zeros((row, col, col))
    for i in range(row):
        threshold = np.sort(np.abs(W[i].flatten()))[int(col * col * 0.1)]
        result[i] = W[i] * (W[i] >= threshold)
    return result
# 权重对角矩阵Ws
def set_Ws(W):
    W = W.detach().numpy()
    row = W.shape[0]
    col = W.shape[1]
    Ws = np.zeros((row, col, col))
    for i in range(row):
        for j in range(col):
            Ws[i, j, j] = sum(W[i][j])
    Ws = torch.FloatTensor(Ws)
    return Ws

def HyperGraphconv(node_S,node_R,Weight,Ws,beta = 0.8):
    Weight = Weight.detach().numpy()
    Ws = Ws.detach().numpy()
    I = node_S * Weight
    count = Weight.shape[0]
    n = Weight.shape[1]
    F = np.ones((n, n))
    E = np.identity(n)
    delta_I = F * (np.matmul(I, node_R))*E
    delta_We = np.matmul(np.matmul(Weight, np.linalg.inv(Ws)), delta_I)
    new_weight_delta = np.zeros((count, n, n))
    for i in range(count):
        new_weight_delta[i] = delta_We[i] + delta_We[i].T
    # 边影响力卷积
    new_Weight = Weight + beta*new_weight_delta
    # 更新权重
    new_Weight = torch.FloatTensor(new_Weight)
    return new_Weight,new_weight_delta

def updata_distant(Weight,Distant):
    new_Distant = Distant -  Weight
    return new_Distant

def set_HT(H):
    H = H.detach().numpy()
    row = H.shape[0]
    col = H.shape[1]
    HT = np.zeros((row, col, col))
    for i in range(row):
        HT[i] = H[i].T
    HT = torch.FloatTensor(HT)
    return HT

def set_invDE(De):
    row = De.shape[0]
    col = De.shape[1]
    invDE = np.zeros((row, col, col))
    temp = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            temp[i, j] = De[i, j, j]
    for k in range(row):
        invDE[k] = np.diag(np.power(temp[k], -1))
    invDE = torch.FloatTensor(invDE)
    return invDE

def set_DV2(Dv):
    row = Dv.shape[0]
    col = Dv.shape[1]
    DV2 = np.zeros((row, col, col))
    temp = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            temp[i, j] = Dv[i, j, j]
    for k in range(row):
        DV2[k] = np.diag(np.power(temp[k], -0.5))
    DV2 = torch.FloatTensor(DV2)
    return DV2





