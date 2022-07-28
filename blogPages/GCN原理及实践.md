# [更新]GCN原理及实践

> by WangYC_99
>
> @ NWPU changan Jan.11st 2022
>
> 第一次更新：@home Feb.21st 2022

## 1. 为什么要引入GCN？

### 1.1 GCN vs CNN or RNN

回忆一下，我们做图像识别，对象是图片，是一个二维的结构，于是人们发明了CNN这种神奇的模型来提取图片的特征。**CNN**的核心在于它的kernel，kernel是一个个小窗口，在图片上平移，通过卷积的方式来提取特征。这里的关键在于图片结构上的**平移不变性**：一个小窗口无论移动到图片的哪一个位置，其内部的结构都是一模一样的，因此CNN可以实现**参数共享**。这就是CNN的精髓所在。

**RNN**系列，它的对象是自然语言这样的序列信息，是一个一维的结构，RNN就是专门针对这些序列的结构而设计的，通过各种门的操作，使得序列前后的信息互相影响，从而很好地捕捉序列的特征。

但是对于图结构来说，图是一种非欧几里得结构的数据，也就是图结构**没有平移不变形**。

因此要引入新的神经网络来对图的结构进行学习。

### 1.2 GCN vs Graph Embedding

Graph Embedding技术一般通过特定的策略对图中的顶点进行游走采样进而学习到图中的顶点的相似性，可以看做是一种**将图的拓扑结构进行向量表示**的方法。

也就是说embedding学习的更多是图的拓扑结构。

当遇到节点带有信息的场景时，基于GraphEmbedding的方法通常是将属性特征拼接到顶点向量中提供给后续任务使用。

而GCN则不同，可以直接通过对图的拓扑结构和顶点的属性信息进行学习来得到任务结果。

## 2. 什么是GCN？

GCN是谱图卷积的一阶局部近似，是一个**多层的图卷积神经网络，每一个卷积层仅处理一阶邻域信息，通过叠加若干卷积层可以实现多阶邻域的信息传递**。

GCN最重要的公式：

![image-20220113190113634](GCN原理及实践.assets/image-20220113190113634.png)

D是度矩阵，A是邻接矩阵，W是要训练的参数矩阵。

其中H的意思可以大致理解为一个层的特征（称为激活单元矩阵，其中第0层的H表示的就是最原本的特征矩阵X），则此公式的意义在于告诉了我们如何从第*l*层的H来计算第*l + 1*层的H）。

σ是非线性激活函数（ReLU，ELU等）。

其中D，A，H(l)都是已知的，W是要训练的参数，初始为随机值，σ可以根据自己的需求进行选择。

**for example：**

设计一个两层的GCN网络，激活函数分别取ReLU和SoftMax，那么网络的整体公式就长这样：

![image-20220113190125918](GCN原理及实践.assets/image-20220113190125918.png)

## 3. 源码分析与实践

### 3.1 源代码目录框架

```
.
├── LICENCE
├── README.md
├── data
│   └── cora
│       ├── README
│       ├── cora.cites
│       └── cora.content
├── figure.png
├── pygcn
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── layers.cpython-36.pyc
│   │   ├── models.cpython-36.pyc
│   │   └── utils.cpython-36.pyc
│   ├── layers.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
└── setup.py
```

### 3.2 dataset

测试数据是一个论文引用的场景。其中cora.cites表示论文之间的引用关系，cora.content表示每篇论文的feature。

### 3.3 网络结构

```python
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
```

这段就是一个非常经典的神经网络结构，init函数中初始化父类和定义自己的函数，forward函数中调用这些函数来完成网络的搭建。

其中的GraphConvolution函数为自己定义的一个模型class：

### 3.4 核心代码部分（公式代码）

以下部分是对应了论文中计算公式的代码部分

```python
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
```

这其中与公式![image-20220113190113634](GCN原理及实践.assets/image-20220113190113634.png)对应的是:

```python
support = torch.mm(input, self.weight)
output = torch.spmm(adj, support)
```

其中torch.mm功能为矩阵相乘，spmm的作用为稀疏矩阵相乘，其中的adj就是邻接矩阵，而且是经过了处理以后的邻接矩阵。

处理的过程在：

```python
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
```

利用了scipy中的多维矩阵运算封装来进行计算。scipy是一个依托于numpy的科学计算的库。

### 3.5 训练效果：

1. 采用原模型训练：

![image-20220114011821081](GCN原理及实践.assets/image-20220114011821081.png)

2. 修改normalize函数为-1/2的：

![image-20220114011943955](GCN原理及实践.assets/image-20220114011943955.png)

可以看出效果有略微地增强。

## 4. 心得体会

graph的网络难点在于graph的数据结构是一个非欧几里得的数据结构，于是我们不直接研究graph的结构，而是转而研究其邻接矩阵以及参数矩阵，由于这些矩阵是典型的欧几里得结构，因此使得网络像以往的图神经网络一样进行。

## Reference

<a herf="https://zhuanlan.zhihu.com/p/78624225">知乎@浅梦 《GCN：算法原理，实现和应用》</a>

<a herf="https://zhuanlan.zhihu.com/p/78624225">知乎@蝈蝈《何时能懂你的心——图卷积神经网络（GCN）》</a>

<a herf="https://zhuanlan.zhihu.com/p/78624225">知乎@MyEncyclopedia《图神经网络 GCN Pytorch 版实现代码讲解》</a>

https://blog.csdn.net/d179212934/article/details/108093614

