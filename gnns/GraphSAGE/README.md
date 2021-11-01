## Chapter7：GraphSage示例（使用Cora数据集）

### 文件说明

| 文件        | 说明                          |
| :---------- | ----------------------------- |
| main.py     | 基于Cora数据集的GraphSage示例 |
| net.py      | 主要是GraphSage定义           |
| data.py     | 主要是Cora数据集准备          |
| sampling.py | 简单的采样接口                |

### 运行示例

```shell
cd chapter7
python3 main.py
```

TODO: 支持在Colab中运行


### 本示例中Cora数据集的内容

#### Cora原内容说明

该数据集共2708个样本点，每个样本点都是一篇科学论文，所有样本点被分为7个类别，类别分别是1）基于案例；2）遗传算法；3）神经网络；4）概率方法；5）强化学习；6）规则学习；7）理论

每篇论文都由一个1433维的词向量表示，所以，每个样本点具有1433个特征。词向量的每个元素都对应一个词，且该元素只有0或1两个取值。取0表示该元素对应的词不在论文中，取1表示在论文中。所有的词来源于一个具有1433个词的字典。

每篇论文都至少引用了一篇其他论文，或者被其他论文引用，也就是样本点之间存在联系，没有任何一个样本点与其他样本点完全没联系。如果将样本点看做图中的点，则这是一个连通的图，不存在孤立点。

#### 本示例中使用的数据说明

项目中使用的数据说明

```python
Node's feature shape:  (2708, 1433)
Node's label shape:  (2708,)
Adjacency's shape:  2708 
Number of training nodes:  140 
Number of validation nodes:  500 
Number of test nodes:  1000 
Cached file: ../data/cora\ch7_cached.pkl
```

**_data文件加数据说明：_**

[Cora 数据集介绍](https://aistudio.baidu.com/aistudio/projectdetail/2246479?shared=1)

ind.cora.x : 训练集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (140, 1433)

ind.cora.tx : 测试集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (1000, 1433)

ind.cora.allx : 包含有标签和无标签的训练节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为：(1708, 1433)，可以理解为除测试集以外的其他节点特征集合，训练集是它的子集

ind.cora.y : one-hot表示的训练节点的标签，保存对象为：numpy.ndarray

ind.cora.ty : one-hot表示的测试节点的标签，保存对象为：numpy.ndarray

ind.cora.ally : one-hot表示的ind.cora.allx对应的标签，保存对象为：numpy.ndarray

ind.cora.graph : 保存节点之间边的信息，保存格式为：{ index : [ index_of_neighbor_nodes ] }

ind.cora.test.index : 保存测试集节点的索引，保存对象为：List，用于后面的归纳学习设置。


