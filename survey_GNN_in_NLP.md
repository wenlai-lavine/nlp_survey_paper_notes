### GNN在NLP中的应用综述论文笔记
1. paper: [Graph Neural Networks for Natural Language Processing: A Survey](https://arxiv.org/pdf/2106.06090.pdf)
2. Authors：Lingfei Wu (京东)
#### 1. Abstract
+ 尽管文本的输入通常是表示为序列的，但是NLP中的一些问题很多是可以用到Graph结构的
+ 三个分类：图的构造、图表示学习、基于图的encoder-decoder模型
#### 2. Introduction
+ NLP在deep learning中通常被表示为序列中的token
+ 然而，有一些NLP问题用图结构表示会有更好的展示。例如：一个文本序列中的句子结构信息(语法树、依存树等)可以作为具体任务的知识引入模型中；
  相似的，语义信息(AMR、知识抽取的图谱)
+ 不幸的是，不管是图像还是文本与Graph还是有一些差距，需要额外的技术去进行转换
+ GNN运用在NLP中受到了一些挑战，主要包括：
    + 怎么将原始的文本数据转换为图结构数据？
    + 怎么去更好的学习图表示？
    + 如何更有效的建模复杂数据？
+ Contributions
  + 对GNN用在NLP中有一个新的分类
  + 对最新的SOTA方法的GNN用在NLP中有一个详细的解释
  + 如何应用也有更好的解释
  + 提供了最新的dataset、evaluation metrics、open-source的code
  + 提出一些挑战和未来研究方向
#### 3. Graph Based Algorithms for Natural Language Processing
从图的角度去梳理NLP中的一些算法
##### 3.1 Natural Language Processing: A Graph Perspective
+ 三种表示自然语言的方式：
  + 1. bag of tokens: 完全忽略了token在句子中的位置，仅仅考虑了token的频率
      + 主题模型(topic modeling)
  + 2. Sequence of tokens: 获取更加丰富的信息
      + 线性CRF、word2vec
  + 3. graph
      + dependency graphs, constituency graphs, AMR graphs, IE graphs, lexical networks, and knowledge graphs.
      + 获取更加丰富的文本之间的关系
      + 一些传统的基于图的模型：random-walk、label-propagation
##### 3.2 Graph Based Methods for Natural Language Processing
对于NLP中基于图的传统算法，可以参考：[Graph-based natural language processing and information retrieval](https://www.cambridge.org/core/books/graphbased-natural-language-processing-and-information-retrieval/216B4D2C3F82BF04C0CC383CD3760C19)
###### 3.2.1 Random Walk Algorithms
随机游走是一类基于图的算法，可在图中生成随机路径。 为了进行随机游走，可以从图中的任何节点开始，并根据一定的转移概率
重复选择每次访问随机相邻节点。 随机游走中所有访问过的节点然后形成一条随机路径。 随机游走收敛后，可以获得图中所有节
点的平稳分布，该分布可用于通过对概率分数进行排序来选择图中具有高结构重要性的最显着节点或测量两个图的相关性，通过计算两个随机游走分布之间的相似性
###### 3.2.2 Graph Clustering Algorithms
常见的图聚类算法包括spectral聚类、随机游走聚类和min-cut聚类。 
spectral聚类算法利用图的拉普拉斯矩阵的谱（特征值）在使用现有算法（如 K 均值）进行聚类之前执行降维。 
随机游走聚类算法通过在图上进行 t 步随机游走来运行，因此，每个节点都表示为概率向量，表示图中所有其他节点的 t 步生成概率。
任何聚类算法都可以应用于生成链接向量。 请注意，出于图聚类的目的，更优选较小的 t 值，因为我们更感兴趣的是捕获局部结构信息
而不是全局结构信息（由随机游走收敛后的平稳分布编码）。 min-cut算法也可用于将图划分为簇。
###### 3.2.3 Graph Matching Algorithms
图匹配算法旨在计算两个图之间的相似性。 其中，Graph Edit Distance是最常用的衡量两个图相异度的方法。
它将距离计算为将一个图转换为另一个图所需的更改次数（即添加、删除、替换）。 然后可以将相异性分数转换为相似性分数。
###### 3.2.4 Label Propagation Algorithms
标签传播算法（LPA）是一类半监督的基于图的算法，它将标签从标记数据点传播到以前未标记的数据点。 
基本上，LPA 通过在图中迭代地传播和聚合标签来运行。 在每次迭代中，每个节点都会根据其相邻节点拥有的标签更改其标签。 
结果，标签信息在图形中扩散。
###### 3.2.5 Limitations and Connections to GNNs
+ 传统图算法的局限性：
  + 1. 表达能力有限,主要关注的是图的结构信息，没有考虑到NLP中节点和边的特征
  + 2. 没有统一的学习框架
+ 怎么去拥有一个统一的基于图的学习框架？对图结构和节点/边的属性有着强大的建模能力？---> 图神经网络
#### 4. Graph Neural Network
+ 4.1 Graph Filtering
  + Spectral-based Graph Filters
  + Spatial-based Graph Filters
  + Attention-based Graph Filters
  + Recurrent-based Graph Filters
+ 4.2 Graph Pooling
  + Graph Pooling 为以图为中心的下游任务生成图级表示，例如基于从图过滤中学习到的节点嵌入的图分类和预测。
  这是因为学习到的节点嵌入对于以节点为中心的任务已经足够了，但是，对于以图为中心的任务，需要整个图的表示。 为此，我们需要总结节点嵌入信息和图结构信息。
  + Flat Graph Pooling
  + Hierarchical Graph Pooling

#### 5. Graph Construction Methods for Natural Language Processing  
  + 如何从文本序列构建图输入？
  + **Static Graph Construction**
    + 静态图构建方法旨在通常通过利用现有的关系解析工具（例如，依赖解析）或手动定义的规则在预处理期间构建图结构。 从概念上讲，静态图结合了隐藏在原始文本序列中的不同领域/外部知识，从而用丰富的结构化信息增强了原始文本
    + 包括以下方法：
      + Dependency Graph Construction
        + 捕捉给定句子中不同目标对象之间的依赖关系
        + Step1: Dependency Relations
        + Step2: Sequential Relations
        + Step3: Dependency Graph
      + Constituency Graph Construction
        + 它能够捕获一个或多个句子中基于短语的句法关系。 与仅关注单个单词（即单词级别）之间的一对一对应关系的依存解析不同，选区解析对一个或多个对应单词的组合（即短语级别）进行建模。 因此，它提供了有关句子语法结构的新见解。
        + Step1: Constituency Relations
        + Step2: Constituency Graph
      + AMR Graph Construction
        + AMR图是有root、label、directed、acyclic graphs，广泛用于表示非结构化自然文本的抽象概念和具体的自然文本之间的高级语义关系。 与句法特性不同，AMR 是高级语义抽象。 更具体地说，语义相似的不同句子可能共享相同的 AMR 解析结果.
        + Step1: AMR Relations
        + Step2: AMR Graph
      + Information Extraction Graph Construction
        + 信息提取图（IE Graph）旨在提取结构信息以表示自然句子（例如基于文本的文档）之间的高级信息。 这些提取的关系捕获了远距离句子之间的关系，已被证明在许多 NLP 任务中很有帮助
        + Step1: Coreference Resolution
        + Step2: IE Relations
        + Step3: IE Graph Constructions
      + Discourse Graph Construction
        + 当候选文档太长时，许多 NLP 任务都会遇到长依赖挑战。 描述两个句子如何在逻辑上相互联系的话语图被证明可以有效地应对此类挑战.
        + Step1: Discourse Relation
        + Step2: Discourse Graph
      + Knowledge Graph Construction
        + 捕获实体和关系的知识图 (KG) 可以极大地促进许多 NLP 应用中的学习和推理。 
          一般来说，KGs 可以根据它们的图构建方法分为两大类。 
          许多应用程序将 KG 视为非结构化数据（例如文档）的紧凑且可解释的中间表示
        + 相关工作真的太多了，如需了解，详细看原论文这一部分的reference
      + Coreference Graph Construction
        + 在语言学中，当给定段落中的两个或多个术语指代同一个对象时，就会发生共指（或共指）。 
          许多工作表明，这种现象有助于更好地理解语料库的复杂结构和逻辑，解决歧义。 
          为了有效地利用共指信息，共指图被构建为显式建模隐式共指关系。
        + Step1: Coreference Relation
        + Step2: Coreference Graph
      + Similarity Graph Construction
        + 相似性图旨在量化节点之间的相似性，广泛用于许多 NLP 任务。
          由于相似图通常是面向应用的，因此我们专注于为实体、句子和文档等各种类型的元素构建相似图的基本过程，而忽略了特定于应用的细节。
          值得注意的是，相似图的构建是在预处理过程中进行的，而不是与剩余的学习系统以端到端的方式联合训练。
        + Step1: Similarity Graph
        + Step2: Sparse mechanism
      + Co-occurrence Graph Construction
        + 共现图旨在捕捉文本中单词之间的共现关系，该关系广泛用于许多 NLP 任务。 
          共现关系描述了两个单词在固定大小的上下文窗口中共同出现的频率，是捕捉语料库中单词之间语义关系的重要特征。
        + Step1: Co-occurrence Relation
        + Step2: Co-occurrence Graph
      + Topic Graph Construction
        + 主题图建立在多个文档之上，旨在对不同主题之间的高级语义关系进行建模
      + App-driven Graph Construction
      + Hybrid Graph Construction
  
  + **Dynamic Graph Construction**
    + 尽管静态图构建具有将数据的先验知识编码到图结构中的优势，但它也有一些局限性。
      + 首先，为了构建性能合理的图拓扑，需要大量的人力和领域专业知识。 
      + 其次，手动构建的图结构可能容易出错（例如，嘈杂或不完整）。 
      + 第三，由于图构建阶段和图表示学习阶段是不相交的，图构建阶段引入的错误无法纠正，可能会累积到后期，从而导致性能下降。 
      + 最后，图构建过程通常仅由机器学习从业者的见解提供信息，并且对于下游预测任务可能不是最佳的。
    + 为了应对上述挑战，最近对用于 NLP 的 GNN 的尝试探索了动态图构建，而无需借助人力或领域专业知识。
      大多数动态图构建方法旨在动态学习图结构（即加权邻接矩阵），并且图构建模块可以与后续的图表示学习模块联合优化，
      最终面向下游任务。此外，为了有效地进行联合图结构和表示学习，已经提出了各种学习范式。
    + **Graph Similarity Metric Learning Technologies**
      + 基于节点属性包含学习隐式图结构的有用信息的假设，最近的工作已经探索将图结构学习问题转化为定义在节点嵌入空间上的相似性度量学习问题。 
        学习到的相似性度量函数稍后可以应用于一组看不见的节点嵌入以推断图结构，从而实现归纳图结构学习。 
        对于部署在非欧几里得域中的数据（如图形），欧几里得距离不一定是衡量节点相似性的最佳指标。 
        已经为 GNN 的图结构学习提出了各种相似性度量函数。 根据所使用的信息源类型，我们将这些度量函数分为两类：
        基于节点嵌入的相似性度量学习和结构感知相似性度量学习。
      + Node Embedding Based Similarity Metric Learning
        + 基于节点嵌入的相似性度量学习基于节点嵌入的相似性度量函数旨在通过计算嵌入空间中的成对节点相似性来学习加权邻接矩阵。
          常见的度量函数包括基于注意力的度量函数和基于余弦的度量函数。
      + Attention-based Similarity Metric Functions
      + Cosine-based Similarity Metric Functions
      + Structure-aware Similarity Metric Learning
    + **Graph Sparsification Technologies**
      + 现实世界场景中的大多数图都是稀疏图。 相似度度量函数考虑任何一对节点之间的关系并返回一个完全连接的图，这不仅计算成本高，而且可能引入噪声，例如不重要的边。
        因此，明确地对学习到的图结构实施稀疏性可能是有益的。
    + **Combining Intrinsic Graph Structures and Implicit Graph Structures**
    + **Learning Paradigms**
      + 大多数现有的 GNN 动态图构建方法由两个关键的学习组件组成：图结构学习（即相似性度量学习）和图表示学习（即 GNN 模块），
        最终目标是学习优化的图结构和关于某些下游预测任务的表示。 
        如何优化两个独立的学习组件以实现相同的最终目标成为一个重要问题。 在这里，我们重点介绍三种代表性的学习范式

#### 6. Graph Representation Learning for NLP
+ 图表示学习的目标是找到一种方法，通过机器学习模型将图结构和属性的信息合并到低维嵌入中
+ 6.1 GNNs for Homogeneous Graphs(同质图)
  + **Static Graph: Treating edge information as connectivity**
    + Converting Edge Information to Adjacent Matrix
    + Node Representation Learning
  + **Dynamic Graph**
    + 旨在与下游任务共同学习图结构的动态图被图表示学习广泛采用
  + **Graph Neural Networks: Bidirectional Graph Embeddings**
+ 6.2 Graph Neural Networks for Multi-relational Graphs
  + 6.2.1 Multi-relational Graph formalization
  + 6.2.2 Multi-relational Graph Neural Networks
    + R-GCN
    + R-GGNN
    + R-GAT
    + Gating Mechanism
  + 6.2.3 **Graph Transformer**
    + R-GAT Based Graph Transformer
    + Structure-aware Self-attention Based Graph Transformer
+ 6.3 Graph Neural Networks for Heterogeneous Graph (异质图)
  + 6.3.1 Levi Graph Transformation
  + 6.3.2 Meta-Path Based Heterogeneous GNN
    + HAN
      + Node-level Aggregation
      + Meta-path Level Aggregation
    + MEIRec
      + Node-level Aggregation
      + Meta-path Level Aggregation
  + 6.3.3 R-GNN Based Heterogeneous GNN
    + HGAT
      + Type-level learning
      + Node-level learning
    + MHGRN
      + k-hop feature aggregation
      + Fusing different relation paths
    + HGT
      + Attention operation
      + Message passing operation
      + Aggregation operation
      + Relative Temporal Encoding

#### 7. GNN Based Encoder-Decoder Models
+ 7.1 Sequence-to-Sequence Models
+ 7.2 Graph-to-Sequence Models
  + Graph-based Encoders
  + Node & Edge Embeddings Initialization
  + Sequential Decoding Techniques
+ 7.3 Graph-to-Tree Models
  + Graph construction
  + Graph encoder
  + Attention
  + Tree decoder
+ 7.4 Graph-to-Graph Models

#### 8. Applications
+ 8.1 Natural Language Generation
  + 8.1.1 Neural Machine Translation (**划重点，细读**)
  + 8.1.2 Summarization
  

  
    
  
    
    
