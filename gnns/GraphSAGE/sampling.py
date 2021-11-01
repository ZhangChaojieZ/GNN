import numpy as np


def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    
    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表
    
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        """
        choice(a, size=None, replace=True, p=None)
        a – 如果是多维数组，则从其元素生成随机样本。 如果是 int，则生成随机样本，就好像 a 是 np.arange(a)
        size - 输出形状。 如果给定的形状是，例如，``(m, n, k)``，则抽取``m * n * k`` 样本。 默认为无，在这种情况下返回单个值。
        replace – 样品是否有更换
        p – 与 a 中每个条目相关的概率。 如果没有给出，样本假设在 a 中的所有条目上均匀分布。
        返回：生成的随机样本
        """
        # 从节点的邻居中进行有放回地进行采样，大小为sample_num的一维数组
        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        # 将生成的采样样本添加到结果列表尾部，维度为src_nodes.size*sample_num的结果
        results.append(res)
    # 将结果折叠成一个一维数组返回，flatten只能作用于numpy对象
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样
    
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id，源节点的id数组
        sample_nums {list of int} -- 每一阶需要采样的个数，是一个数组
        neighbor_table {dict} -- 节点到其邻居节点的映射
    
    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """

    # 转化为数组，[1,2,3]转化为[array([1,2,3], dtype=int64)]，将源节点放在第0个位置
    sampling_result = [src_nodes]
    # 多阶采样，enumerate函数返回元组列表[(下标0,元素值0),(下标1,元素值1),(下标2,元素值2)]
    for k, hopk_num in enumerate(sample_nums):
        # 对第K层进行采样
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        # 将每一层采样的结果拼接起来
        sampling_result.append(hopk_result)
    # 返回一个维度为sample_num.size*hopk_num的数组
    print("sample的多层采样结果",sampling_result)
    return sampling_result
