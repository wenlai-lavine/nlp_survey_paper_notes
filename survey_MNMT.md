### MNMT综述论文笔记
1. paper: [A Comprehensive Survey of Multilingual Neural Machine Translation](https://arxiv.org/abs/2001.01115)
2. Authors：RAJ DABRE, CHENHUI CHU (日本NICT)
#### 1. MNMT分类
+ Multiway Translation
    + 目标：构造一个一对多、多对一、多对多的系统
    + 前提：存在多种语言之间的平行语料
+ Low Resource Translation
    + 利用rich-resource的翻译去提高low-resource的翻译 --> Transfer Learning
    + 利用中间语言 --> Pivot-based translation
+ Multi-Source Translation
    + 现有的多语言翻译句对可以在将来翻译为其他目标语言

#### 2.本文试图去解答的两个问题
+ 是否存在一个模型同时适用于多种语言的翻译？
+ 共享多语言的分布表示是否可以帮助到low-resource languages的翻译

#### 3. Multiway Translation
+ 前提：我们存在所有语言对之间的平行语料
+ 训练目标：最大化所有语言对之间的平均翻译似然
+ 涉及到的几个问题：词汇表和相关的embedding、可堆叠的层(RNN/CNN/FNN)、参数共享、Training protocols、语言差异性问题
  + 参数共享问题(共享参数的程度)：
    + Minimal Parameter Sharing
      + 1.共享一个attention
        + paper: Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism [NAACL2016]
        + idea：模型由每种语言的embedding、encoder和decoder组成，这些模型都共享一个attention。 此外还有两个共享组件：一个用于所有encoder的层，用于通过使用最终encoder状态来初始化初始decoder状态，以及一个用于所有decoder的仿射层，用于在计算softmax之前映射到最终decoder状态。 然而，重点是共享attention
        + 细节：attention中的计算方式与两种语言有所区别，必须兼顾多种语言，具体公式参考原论文
        + 优点：当目标语言是英语的时候，可以超过很多bilingual的model
        + 缺点：模型有大量的参数(模型参数量随语言的数量线性增加)，现实research中很少使用到
    + Complete Parameter Sharing
      + 1. 完全共享(embedding、encoder、decoder、attention)
        + paper：Google’s Multilingual Neural Machine Translation System: Enabling Zero-ShotTranslation [TACL2017]
        + idea：首先，为所有语言生成一个共有词表；其次，所有语言的语料都被合并在一起，唯一的不同是在源语言端加上目标语言指向
      + 2. 每种语言之间的词表是分离的，可以加速inference
        + paper: Toward Multilingual Neural Machine Translation with Universal Encoder and Decoder [IWSLT2016]
        + 优点：这种方法在语言相似的语言对之间会有好处，
        + 缺点：没有说明什么时候share，也没有说明为什么分离词表可以提升性能？
      + 3. 相关技术
        + lexical similarity (词表转换、共享词表)
        + self-attention的框架在MNMT的大部分case中都优于其他框架
        + massively multilingual NMT 的想法将完全共享的idea推向了极限，用最小的参数训练最优的模型
          + 优点：倾向于到英语的翻译，来自英语的翻译效果稍差
          + 缺点：模型的capacity瓶颈
    + Controlled Parameter Sharing (**敲黑板，敲黑板！！！**)
      + 1. 共享encoder，分离各个语言的decoder
        + 原因：generate是在decoder端的，attention也是在decoder端，应该区别对待
        + paper1: Blackwood et al. [Multilingual Neural Machine Translation with Task-Specific Attention.](http://aclweb.org/anthology/C18-1263)   
            + share 目标语言的attention比其他attention恭喜设置更加有效
        + paper2: Sachan and Neubig. [Parameter Sharing Methods for Multilingual Self-Attentional Translation Models](http://aclweb.org/anthology/W18-6327)
            + 比较了多个不同的share策略，发现：共享decoder的self-attention，以及encoder-decoder的cross-attention对于语言学上的不相似语言也同样很有用，通过share self-attention以及cross-attention，目标语言的表示会与源语言有更好的对齐
        + paper3: Wang et al., [A Compact and Language-Sensitive Multilingual Translation Method](https://doi.org/10.18653/v1/P19-1117)
            + 提出了一个机制去生成一个通用的表示
        + paper4： Bapna and Firat [Simple, Scalable Adaptation for Neural Machine Translation](https://doi.org/10.18653/v1/D19-1165)
            + 提出用一个language-specific的层去fine-tune我们的MNMT模型
      + 2. routing-select
        + 有一些参数是不必共享的
        + paper1：Zaremoodi et al. [Adaptive Knowledge Sharing in Multi-Task Learning: Improving Low- Resource Neural Machine Translation](http://aclweb.org/anthology/P18-2104)
          + 训练一个参数选择网络动态的选择哪些参数应该共享
        + paper2：Platanios et al. [Contextual Parameter Generation for Universal Neural Machine Translation](http://aclweb.org/anthology/D18-1039)
          + 从训练数据集中学习参数共享的程度
  + 语言差异(Divergence)问题
    + background：MNMT中的核心任务就是如何在不同语言之间对齐词和句子的表示，这就涉及到多语言模型中的表示学习问题。
    + 1. 多语言表示的实质(nature)
      + 一些可视化多语言模型的工作发现，多语言模型的encoder在相似语言之间学习了相似的表示，因为这些可视化是在很低纬度发现的，可能不够准确
      + [Kudugunta et al.](Investigating Multilingual NMT Representations at Scale) 用 [SVCCA](https://github.com/google/svcca) 的方法系统的学习了多语言模型中的表示。它们的工作发现：
        + encoder的表示通常会在相似的语言对之间产生，这也间接的解释了为什么迁移学习在相似语言对中表现更好
        + encoder与decoder之间的边界并不清楚，源语言取决于目标语言，反之亦然
        + 表示相似性因层而异。在encoder的更高层中，不变性增加。 另一方面，不变性在decoder端的较高层中降低。 这也是很好解释的，因为decoder对要生成的目标语言很敏感。 decoder必须在语言不可知表示和语言感知表示之间取得适当的平衡。
    + 2. encoder端的表示
        + 有两个原因可能会导致encoder会更加语言独立(language dependent):
          + 来自不同源语言的平行句对可能有不同的标记，因此decoder的注意力机制可以看到不同语言的等效句子的可变数量的encoder表示
        + 几个相关工作：
          + 1. 生成固定的上下文表示引入attention网络中去
          + 2. 对所有语言又一个单一的encoder，decoder-specific
          + 3. 对输入句子进行重排序以减少语言之间的差异性
  + 3. Decoder端的表示
      + background: 当涉及多种目标语言时，需要解决decoder表示的分歧。 这是一个具有挑战性的场景，因为decoder应该生成表示，以帮助它在每种目标语言中生成有意义且流畅的句子。 因此，学习语言不变表示和能够生成语言特定翻译之间的平衡至关重要
      + 1. language tag 方法
        + 句子开头/句子结尾
      + 2. 解决language divergence最好的办法是源语言共享一个encoder，目标语言有不同的decoder
      + 3. 先将语言进行聚类的一些工作
  + 4. language tag的一些影响
    + [wang et al.,](ThreeStrategiestoImproveOne-to-ManyMultilingual Translation) 探讨不同的语言tag
    + 通过预训练的一些方法
  + Training Protocols问题 （训练策略问题）
    + 1. Single Stage Parallel/ Joint Training
      + 上采样数据小的语言对，temperature-based
    + 2. Multi-stage Training
      + Knowledge Distillation
      + Incresmental Training (unseen language)

#### 4. MNMT for Low-Resource Languages  
  + 1. Training
    + 大部分的方法探讨在源端利用 transfer learning 方法：high-resource 与 low-resource share相同的目标语言
      + 最简单的方法：**Joint Training** [Google’s Multilingual Neural Machine Translation System: Enabling Zero-ShotTranslation] ，但是最终的模型可能不好tune
      + 一个更好的方法：用子语言对去**fine-tune**父模型：
        + paper: Zoph et al. [Transfer Learning for Low-Resource Neural Machine Translation](https://doi.org/10.18653/v1/D16-1163)
          + idea: 首先，在high-resource语言对中训练一个parent model，child model利用父模型的参数去初始化low-resource语言对
        + paper: Meta Learning [Meta-Learning for Low-Resource Neural Machine Translation](http://aclweb.org/anthology/D18-1398)
          + idea: 将子语言对考虑在内，提供了更加简单的fine-tune策略，得出结论：更多的父语言对模型会提升子语言对的性能
  + 2. Lexical Transfer
    + 随机初始化、把monolingual的embedding map到同一个space
  + 3. Syntactic Transfer
    + 
  + 4. Language Relatedness
    + 

#### 5. MNMT for unseen language pairs
  + 1. Pivot Translation
    + 传统的三种解决方案
    + 变种：从S-P系统中抽取n-best结果，P-T可以生成m-best结果，最后n*m个结果可以re-rank
  + 2. Zero-shot Translation
    + MNMT作为隐形的枢轴系统
    + 1. Limitations of Zero-shot Translation
      + 有文章指出：训练阶段更多的包含更多的语言更有利于zero-shot的翻译
      + zero-shot的结果一般都会低于pivot-based translation
      + 可能存在两个局限：
        + Spurious Correlation between input and output language (输入和输出语言之间的虚假相关性): 有文章指出，当评估正确的语言生成译文时，zero-shot的性能可以达到pivot-based的性能
        + Language variant encoder representation (语言多样性的encoder表示)：源语言和pivot表示之间的不一致性
    + 2. Improving Zero-shot Translation 
        + Minimize divergence between encoder representations: 在训练阶段，额外的目标确保源语言和枢轴encoder表示尽可能的相似（无监督的方法）
        + Encourage output agreement：借助一些辅助语言
        + Effect of corpus size and number of langugaes: 增加语言数量
        + Addressing wrong language generation：在输出的时候filter掉不相关的语言
  + 3. Zero-resource Translation
    + 训练的时候，我们指定unseen langugae去训练一个对unseen更加language-specific的model
    + 1. Synthetic Corpus Generation: 利用back-translation在pivot-language之间生成伪语料
    + 2. Iterative approaches: time-consuming
    + 3. Teacher-Student Training: 
    + 4. Combining Pre-trained encoders and decoders

#### 6. Multi-Source NMT
+ 定义：如果语言源语言已经被翻译成很多个目标语言，那么这些平行句对可以一起提升目标语言的翻译性能，这个技术成为Multi-source NMT
+ Why Multi-source MT? 欧盟中多种使用语言的应用场景
+ Missing Source Sentences
+ Post-Editing
  
  
  