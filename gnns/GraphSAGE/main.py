# coding: utf-8
"""
基于Cora的GraphSage示例
"""
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import GraphSage
from data import CoraData
from sampling import multihop_sampling

from collections import namedtuple

INPUT_DIM = 1433  # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 7]  # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 16  # 批处理大小
EPOCHS = 20  # 训练次数
NUM_BATCH_PER_EPOCH = 20  # 每个epoch循环的批次数
LEARNING_RATE = 0.01  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 定义一个namedtuple类型Data，并包含'x', 'y', 'adjacency_dict','train_mask', 'val_mask', 'test_mask'属性。
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

# 获得数据
data = CoraData().data
# 归一化数据，使得每一行和为1，形状shapex: (2708, 1433)
x = data.x / data.x.sum(1, keepdims=True)

# where选取出括号内为True的元素，训练集的索引数组
# data.train_mask 是 bool类型的数组,np.where()为true的输出下标,不加[0]的话是 数组的 数组 即(数组,) 所以要得到数组 取[0]
# print(np.where(data.train_mask))
train_index = np.where(data.train_mask)[0]
# 标签
train_label = data.y
# 测试集起始索引
test_index = np.where(data.test_mask)[0]
# 生成模型
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
# 打印模型
print(model)
# nn.CrossEntropyLoss()使用交叉熵损失，.to(DEVICE)表明复制一份数据到DEVICE上面
criterion = nn.CrossEntropyLoss().to(DEVICE)
"""
为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
要构建一个优化器optimizer，你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。 
然后，可以指定程序优化特定的选项，例如学习速率，权重衰减等。
"""
# 优化器，使用了Adam算法，weight_decay (float, 可选) – 权重衰减
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)


def train():
    # 训练模式，训练用train()，测试用eval()
    model.train()
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            # 获取批处理的这一批数据Batch，是一个索引数组，源节点数组
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            # 批处理数据batch的标签，所以数组
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            # 采样的结果，素所有K阶节点的采样结果都在这个数组中，这其中第0个元，是所有原始节点的数组
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
            print("多层采样结果", batch_sampling_result)
            # 对采样结果进行一个转化，转化为tensor
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
            # print("节点特征",batch_sampling_x)
            # 给模型传入参数，源节点以及源节点所有k阶邻居节点的特征
            batch_train_logits = model(batch_sampling_x)
            print("每个batch训练之后的节点特征", batch_train_logits)
            # 计算损失
            loss = criterion(batch_train_logits, batch_src_label)
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播计算参数的梯度
            loss.backward()
            # 使用优化方法进行梯度更新
            optimizer.step()
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()


def test():
    # 切换到测试模式
    model.eval()
    with torch.no_grad():
        # 测试集采样
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        # 测试集采样节点的特征矩阵
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        # 使用训练好的网络进行测试
        test_logits = model(test_x)
        # 测试集的标签
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        # 从学习的结果中获取最大概率对应的标签作为预测值
        predict_y = test_logits.max(1)[1]
        # 计算预测的准确率
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)


if __name__ == '__main__':
    train()
