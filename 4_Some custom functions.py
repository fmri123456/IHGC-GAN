import numpy as np

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    return acc

def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get, labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot
# 风险预测
def pred(target_H, generator_H):  # 将生成的关联矩阵与目标关联矩阵作比较
    target_H = target_H.detach().numpy()
    generator_H = generator_H.detach().numpy()
    count = generator_H.shape[0]
    n_edge = generator_H.shape[2]
    similars = np.zeros((count, n_edge))
    for i in range(count):
        for j in range(n_edge):
            temp = np.sum(target_H[0, :, j] == generator_H[i, :, j])
            similars[i, j] = temp / n_edge
    similar = similars.mean(axis=1)  # 对行求平均值
    return similar
