import torch
# torch.nn包内就是类似这种高级封装，里面包括了卷积，池化，RNN等计算,以及其他如Loss计算，可以把torch.nn包内的各个类想象成神经网络的一层
import torch.nn as nn
import torch.nn.functional as F
# 参数初始化方式的一个包
import torch.nn.init as init


# 这里表明NeighborAggregator继承自nn.Module，只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
# 邻居聚合操作
class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居
            对邻居进行聚合，重载了Module类的__init__函数和forward函数

        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        # torch.Tensor是torch.FloatTensor的简称，
        # torch.Tensor(input_dim, output_dim)表示创建一个维度为input_dim*output_dim的tensor，默认类型为float
        """
        nn.Parameter含义是将一个固定不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的)，所以经过类型转换这个weight变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化.
        """
        #  这里是将把weight转化成可以训练的参数，初始是一个维度为input_dim*output_dim的初始矩阵
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # 如果使用偏执，则将偏执也转化为一个可以训练的向量，维度为output_dim
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        # 初始化weight和bias参数，上述相当于定义一个变量
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming均匀分布方式进行初始化
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            # 全0分布初始化
            init.zeros_(self.bias)

    """
    输入neighbor_feature表示需要聚合的邻居节点的特征，它的维度为N_src×N_neighbor×D_in，其中N_src表示源节点的数量，N_neighbor表示邻居节点的数量，
    D_in表示输入的特征维度。将这些邻居节点的特征经过一个线性变换得到隐层特征，
    这样就可以沿着第1个维度进行聚合操作了，包括求和、均值和最大值，得到维度为N_src×D_in的输出。
    """

    def forward(self, neighbor_feature):
        # 选择聚合邻居节点的操作，平均值、求和、最大值，对应伪代码11行
        if self.aggr_method == "mean":
            # dim表示对哪个维度进行操作，最外层是第0个dim，最内层是第n个dim，参考：https://blog.csdn.net/qq_23262411/article/details/100398449
            # 这里对dim=1的维度计算，是对所有邻居节点进行聚合
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            # 错误提示
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        """
        当进行向量的内积运算时，可以通过np.dot()
        当进行矩阵的乘法运算时，可以通过np.matmul()或者@
        当进行标量的乘法运算(哈达玛积）时，可以通过np.multiply()或者*
        """
        # matmul是numpy中的运算，这里是聚合之后的结果和权重矩阵相乘，得到隐藏层的输出，对邻居节点聚合后的结果进行一次线性变换
        # 对应伪代码12行对邻居节点聚合后的特征进行线性变换
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        # 如果使用了偏置，则加上偏置
        if self.use_bias:
            neighbor_hidden += self.bias
        # 获得邻居节点聚合之后的隐层特征，y=Wh+b
        return neighbor_hidden

    # 如果要打印用户定制化的额外信息，需要在神经网络模块中重新实现这个方法. 单行字符串或多行字符串都可以用在这个函数中.
    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


# 基于聚合后的结果更新中心节点特征
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN层定义，重载了Module类的__init__函数和forward函数
        将邻居节点聚合的特征与经过线性变换的中心节点的特征进行求和或者级联

        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        # 添加断言，确保邻居聚合方法和中心节点聚合方法正确
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        # 使用聚合函数对邻居节点特征进行聚合
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        # 定义权重
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        # 获得邻居节点聚合之后的特征，这里不需要再和权重相乘了，前面对邻居节点进行聚合是已经相乘过了
        # 这里self.aggregator(neighbor_node_features)是把neighbor_node_features作为参数传递给NeighborAggregator的forward函数，并获取执行后的结果
        neighbor_hidden = self.aggregator(neighbor_node_features)
        # 这里需要对原始中心节点进行一次线性变换，邻居节点聚合后的特征和中心节点特征两次线性变化的权重不同
        self_hidden = torch.matmul(src_node_features, self.weight)

        # 判断邻居节点聚合的特征和中心节点的特征是求和操作还是拼接操作，对应伪代码12行
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        # 判断是否使用激活函数
        if self.activation:
            # 返回经过激活函数之后的结果，节点的特征矩阵
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


# 基于前面定义的采样和节点特征更新方式，就可以实现的计算节点嵌入的法。下面定义了一个两层的模型，隐藏层节点数为64，假设每阶采样节点数都为10，
# 那么计算中心节点的输出可以通过以下代码实现。其中前向传播时传入的参数node_features_list是一个列表，其中第0个元素表示源节点的特征，其后的元素表示每阶采样得到的节点特征。
# 在采样时就将采样结果压缩成了一维数组，所以第0个元素之后就是源节点的采样得到的节点
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        """
        ""GraphSage层定义，重载了Module类的__init__函数和forward函数
        得到节点的embedding

        Args:
            input_dim: 输入特征的维度
            hidden_dim: 输出的维度，这里hidden_dim是一个列表，表示不同层（下表对应第k层）的隐层特征的维度
            num_neighbors_list: 每阶采样邻居的节点数，是一个一维数组，下标对应第k层，元素值对应该层采样的节点数
            """
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 得到每层的采样节点数数组
        self.num_neighbors_list = num_neighbors_list
        # 层次采样结点数数组的长度对应层数
        self.num_layers = len(num_neighbors_list)
        # nn.ModuleList()是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器，它有类似list的append等方法
        # 将参数添加到网络中，才能使这些参数在训练的过程中进行更新
        self.gcn = nn.ModuleList()
        # 将第一层网络添加到ModuleList中
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        # 将后续的网络层存到ModuleList中，对大于2层的网络有效。这里实际是2层，暂时没用到。
        for index in range(0, len(hidden_dim) - 2):
            # 网络是连续的，index对应的是上一层的输出维度，作为这一次的输入维度。依次循环
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        # 将最后一层网络添加到ModuleList中
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
        # print("hidden_dim的内容",hidden_dim)

    def forward(self, node_features_list):
        # node_features_list是一个列表，其中第0个元素表示源节点（不止一个）的特征，其后的元素表示每阶采样得到的节点特征。
        # node_features_list[0]是所有源节点的特征，node_features_list[1]是node_features_list[0]中节点的采样节点的特征
        hidden = node_features_list
        # 聚合操作的层数，这里设置的l=2
        for l in range(self.num_layers):
            next_hidden = []
            # 选取不同层对应的SageGCN层
            gcn = self.gcn[l]
            """
            这理的内层循环是：假设num_layers=3，当l=0时，内层会进行3次循环hop=0,1,2，输入是node_features_list，这表明gcn[0]这一层网络模型，会从最底层逐层向上聚合节点特征，更新隐层特征hidden，这里len(hidden)=3;
            l=1时，内层进行2次循环hop=0,1，输入则为上一层循环的结果next_hidden，使用新的gcn[1]网络，更新hidden，这里len(hidden)=2；逐层缩小hidden，直到hidden剩下一个元素，那么hidden[0]就是源节点对应的聚合之后的特征
            """
            for hop in range(self.num_layers - l):
                # 首先获得源节点的特征
                src_node_features = hidden[hop]
                # 源节点的个数
                src_node_num = len(src_node_features)
                # 邻居节点特征，将邻居节点特征矩阵进行一个转化，
                neighbor_node_features = hidden[hop + 1].view((src_node_num, self.num_neighbors_list[hop], -1))
                # 得到每层聚合之后新的节点特征
                h = gcn(src_node_features, neighbor_node_features)
                # 将每一层的聚合特征添加到数组中
                next_hidden.append(h)
            # 更新输入，用于下一层网络训练
            hidden = next_hidden
        # 原始节点已经更新完成，第0个位置对应的就是原始节点更新之后的新的节点的特征
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
