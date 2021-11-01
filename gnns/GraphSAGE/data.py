import os
import os.path as osp
import pickle
import numpy as np
import itertools
import scipy.sparse as sp
import urllib
from collections import namedtuple

# 定义一个namedtuple类型Data，并包含'x', 'y', 'adjacency_dict','train_mask', 'val_mask', 'test_mask'属性。
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="../data/cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency_dict: 邻接信息，，类型为 dict
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: ../data/cora
                缓存数据路径: {data_root}/ch7_cached.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        # 数据的根路径
        self.data_root = data_root
        # 缓存数据
        save_file = osp.join(self.data_root, "ch7_cached.pkl")
        # 如果缓存数据存在且不需要重建，则直接加载缓存文件中的数据；否则，重新加载数据，并缓存到缓存文件中
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        # 读取数据
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        """
        数据说明见README.md
        """
        # print("tx:",tx)
        # print("allx:",allx)
        # print("y:",y)
        # print("ty:",ty)
        # print("ally:",ally)
        # print("graph:",graph)
        # print("test_index:",test_index)

        # [0,...139] 140个元素，训练集的索引数组
        train_index = np.arange(y.shape[0])
        # [140 - 640] 500个元素，验证集的索引数组
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        # 对test_index测试集的索引数组进行排序
        sorted_test_index = sorted(test_index)

        # allx和tx拼接，1708 +1000 =2708，得到所有节点的特征向量
        x = np.concatenate((allx, tx), axis=0)
        # 节点的标签标签向量
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        # 打乱顺序，单纯给test_index中元素值对应位置的数据排个序,不清楚具体效果
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        # 节点的个数
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        # 前140个元素为训练集，以train_index数组中的值为下标，将对应位置的元素值置为True
        train_mask[train_index] = True
        # 140 -639 500个验证集
        val_mask[val_index] = True
        # 1708-2708 1000个元素测试集
        test_mask[test_index] = True
        # 邻接表表示图，图是以字典的形式存在的
        adjacency_dict = graph
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency_dict=adjacency_dict,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out
