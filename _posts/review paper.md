1. ##### A residual convolutional long short-term memory neural network for accurate cutterhead torque prediction of shield tunneling machine 

在掘进过程中，由于地质条件的变化，对盾构机的运行参数进行动态调整至关重要。刀盘扭矩是关键的负载参数之一，其准确预测可以提前调整刀盘转速和掘进速度等操作参数，避免潜在的刀盘卡钻。本文提出了一种新的残差卷积长短期记忆神经网络（RCLSTMNN），它可以同时利用工作参数和状态参数对盾构机刀盘扭矩进行精确预测。在相关分析的基础上，利用余弦相似性选取对刀盘扭矩影响较大的参数，显著降低了输入维数。采用卷积神经网络（CNN）和长短时记忆（LSTM）提取隐式特征和序列特征，利用残差网络模块避免多层网络训练中梯度消失，提高回归性能。在15个不同的数据集上，与基于机器学习的预测方法（如支持向量机、随机森林和xgboost）和基于深度学习的方法（如CNN、递归神经网络（RNN）、LSTM）进行了比较，以验证该网络的有效性和优越性。结果表明，该模型的预测精度最高可达98.1%，平均预测精度约为95.6%，在大多数情况下优于其他数据驱动模型。此外，用该预测模型预测的刀盘扭矩曲线与实际曲线吻合得更好。

**Comment:** 

1） 利用数据驱动的机器学习方法，而不是基于岩土力学，数值计算，物理模型计算的预测方法。	

2） 输入： 运行参数，状态参数 ， 历史扭矩     输出 ：未来扭矩    决策： 合理的掘进速度，刀盘转速。

3） CNN+LSTM+Residual , 文中大量介绍深度学习基础 ，纯应用。

![](D:\00_code\blyucs.github.io\images\review\dungouji0.png)

![](D:\00_code\blyucs.github.io\images\review\dungouji1.png)

2. #### A novel convolutional neural network with attentive kernel residual learning for feature learning of gearbox vibration signals

振动信号广泛应用于机械健康监测和故障诊断。深层神经网络（DNNs）以卷积神经网络（CNNs）为例，在机械故障诊断中具有很好的特征学习能力。然而，卷积核宽度的设置仍然是一个具有挑战性的问题。虽然剩余学习降低了CNN训练的难度，但跳跃连接可能会将无效特征转移到深层。本文提出了一种新的DNN-注意核残差网络（AKRNet），用于振动信号的特征学习。首先，利用不同核宽的多分支提取振动信号的多尺度特征。其次，提出了一种有选择地融合多通道特征的核选择方法。第三，提出了一种注意残差块来提高特征学习性能，不仅可以减轻梯度消失，而且可以进一步增强特征映射中的脉冲特征。最后，在两台齿轮箱试验台上验证了AKRNet对振动信号特征学习的有效性。实验结果表明，AKRNet对振动信号具有良好的特征学习能力。它在齿轮箱故障诊断中的性能优于其它典型的dnn，如堆叠式自动编码器（SAE）、一维CNN（一维CNN）和残差网络（ResNet）。

![](D:\00_code\blyucs.github.io\images\review\gearbox0.png)

**Comment：**

1）通道注意力机制 + SENET bottleneck

![](D:\00_code\blyucs.github.io\images\review\gearbox1.png)





3. #### Discriminative manifold random vector functional link neural network for rolling bearing fault diagnosis 

![](D:\00_code\blyucs.github.io\images\review\bear.png)



RVFLNN模型 + soft label matrix +   *流形学习*方法(Manifold Learning)   = **DMRVFLNN**

一、宽度学习的前世今生
宽度学习系统（BLS） 一词的提出源于澳门大学科技学院院长陈俊龙和其学生于2018年1月发表在IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS，VOL. 29, NO. 1 的一篇文章，题目叫《Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture 》。文章的主旨十分明显，就是提出了一种可以和深度学习媲美的宽度学习框架。

为什么要提出宽度学习？ 众所周知，深度学习中最让人头疼之处在于其数量庞大的待优化参数，通常需要耗费大量的时间和机器资源来进行优化。

宽度学习的前身实际上是已经被人们研究了很久的随机向量函数链神经网络 random vector functional-link neural network (RVFLNN)，如图所示：

![](D:\00_code\blyucs.github.io\images\review\BLS0.png)

咋一看这网络结构没有什么奇特之处，其实也对，就是在单层前馈网络（SLFN）中增加了从输入层到输出层的直接连接。网络的第一层也叫输入层，第二层改名了，叫做增强层，第三层是输出层。具体来看，网络中有三种连接，分别是

（输入层 => 增强层）加权后有非线性变换
（增强层 => 输出层）只有线性变换
（输入层 => 输出层）只有线性变换
所以在RVFLNN中只有增强层 是真正意义上的神经网络单元，因为只有它带了激活函数，网络的其他部分均是线性的。下面我们将这个网络结构扭一扭：

![](D:\00_code\blyucs.github.io\images\review\BLS1.png)

当我们把增强层和输入层排成一行时，将它们视为一体，那网络就成了由 A（输入层+增强层）到 Y 的线性变换了！线性变换对应的权重矩阵 W 就是 输入层加增强层 到 输出层 之间的线性连接！！

这时你可能要问：那输入层到增强层之间的连接怎么处理/优化？我们的回答是：不管它！！！ 我们给这些连接随机初始化，固定不变！

如果我们固定输入层到增强层之间的权重，那么对整个网络的训练就是求出 A 到 Y 之间的变换 W，而 W 的确定非常简单：
$W=A^{-1}Y$

输入 X 已知，就可以求出增强层 A；训练数据的标签已知，就知道了 Y。接下来的学习就是一步到位的事情了。

为什么可以这样做？
深度学习费了老劲把网络层数一加再加，就是为了增加模型的复杂度，能更好地逼近我们希望学习到的非线性函数，但是不是非线性层数越多越好呢？**理论早就证明单层前馈网络（SLFN）已经可以作为函数近似器了**，可见增加层数并不是必要的。**RVFLNN也被证明可以用来逼近紧集上的任何连续函数，其非线性近似能力就体现在增强层的非线性激活函数上，只要增强层单元数量足够多，要多非线性有多非线性**！

关于RVFLNN，宽度学习的介绍[为什么要做深度学习而不是宽度学习？](https://blog.csdn.net/pengchengliu/article/details/89393016 "With a Title"). 
[宽度学习（Broad Learning System）](https://blog.csdn.net/itnerd/article/details/82871734 "With a Title"). 



#### Research On Fault Prediction Model Based On 5G Data Center

LR

GBDT

Bagging algorithm





#### A Bioinformatic Variant Fruit Fly Optimizer for Tackling Optimization Problems 

果蝇优化算法是一种基于群体的算法，其灵感来源于自然界的真实现象。它因其简洁、结构简单而受到广泛关注。然而，目前对许多实际问题的求解仍可能过于简单化和不尽如人意，其核心倾向在收敛速度和准确率方面还有待进一步发展。因此，针对上述不足，本研究提出了一种品质较好的果蝇优化算法BSSFOA，并对其进行了验证。与果蝇优化算法相比，BSSFOA有两个额外的策略：蝙蝠声纳策略增强探索性，混合分布结合高斯分布和学生分布来提高利用率。根据蝙蝠启发算法，个体果蝇利用蝙蝠声纳策略模拟蝙蝠搜索全局最优解。混合分配机制也提高了果蝇优化算法的开发贴近度。基于30个基准函数的综合测试结果表明，与其他算法相比，所开发的BSSFOA具有更好的性能。此外，还验证了该方法对不同数据集的最优特征选择问题。结果和观察结果生动地揭示了所开发的机制在缓解果蝇优化算法核心问题方面的建设性影响。

huilin chen  审稿较多  --温州大学   --- 已**邀请**



####  **Developing A Deep Learning model for Travel speed prediction using decomposition based smooth time series** 

印度   -- 找不到所说的文章。   -- - 已**reject**



####  Aligning Social Network with Dynamics: A Deep Learning Approach 

社会网络比对是社会网络分析中的一个基本问题，它的目的是在不同的社会网络中识别属于同一个人的社会账户。现有的研究大多集中在静态网络的对准上。然而，社交网络是动态发展的。我们观察到，这样的动态可以揭示更多的区别模式，从而有利于社会网络联盟。这一观察促使我们在动态场景中重新思考这个问题。因此，我们建议利用社交网络的动态特性，设计一个深度架构来解决动态的社交网络对齐问题，称为DeepDSA

----北邮 GRU  还行， 用GRU 做社交网络对齐

--- 拒掉。



####  A sequence-based deep learning approach to predict CTCF-mediated chromatin loop 

染色体的三维结构对转录调控和DNA复制至关重要。各种高通量染色体构象捕获方法已经揭示了CTCF介导的染色质环是3D结构的主要组成部分。然而，CTCF介导的染色质环具有细胞类型特异性，大多数染色质相互作用捕获技术耗时费力，限制了其在大量细胞类型上的应用。基于基因组序列的计算模型非常复杂，足以捕捉染色质结构的重要特征，并有助于识别染色质环。在这项工作中，我们发展了一个卷积神经网络模型Deep-loop

---染色体的机构预测 ，用deep learning 的方式来做的。   一般， **拒掉**， UESTC 



####  Open Set Graph Convolutional Network with Unknown Class Number 

宁波大学

图形卷积网络（GCN）作为图形数据处理的有力工具，在许多机器学习和计算机视觉任务中得到了广泛的应用。然而，现有的gcn通常假设数据集是自包含的，其中严格假设预测的标签包含在训练数据集中。这通常违背了现实世界是开放集的事实，在这种情况下，测试数据的预测标签可能不包含在训练数据集中，并且类号未知。为了克服这些问题，提出了一种新的两级贝叶斯标签生成过程，将GCN模型扩展到贝叶斯框架中，并与聚类方法相结合。然后，针对传统生成模型无法处理实际样本复杂分布的局限性，提出并集成了一种基于深度学习的聚类方法。最后，当模型对聚类数没有初步了解时，我们假设实类是无穷大的，并将Dirichlet过程（DP）与贝叶斯模型相结合来估计实类数。虽然后验推理比较困难，但是我们的模型提供了一种有效的基于变分推理的优化方法。在各种数据集上的实验验证了我们的理论分析，并证明我们的模型能够达到最先进的性能。

---- **送审**   图网络



####  Binary Chimp Optimization Algorithm (BChOA): A new binary meta-heuristic for solving optimization problems 

伊朗， 黑猩猩算法  ， 管 --- 可以**送审**   ---reject

 **Dear Editor：**
The innovation of this paper in the field of optimization method is limited, and there is still room for improvement in theoretical analysis, writing details and experiments。So this article is unsuitable for publication in this journal. 

 **Dear author:**
Sorry to inform you that our journal has recently received so many manuscripts, so we can only consider publishing the most innovative and enlightening articles.Thanks. 





####  An efficient binary Gaining Sharing Knowledge based optimization algorithm with population reduction for feature selection problems 

伊朗  ， reject 

**Dear Editor：**
The innovation of this paper in the field of optimization method is limited. Much length of the article is devoted to the experimental data. Lack of basic theoretical derivation and analysis. So this article is unsuitable for publication in this journal. 



**Dear author:**
Recently we received so many manuscripts, so we can only consider publishing the most innovative and enlightening articles. Much length of the article is devoted to the experimental data. Sorry to inform you that it is suggested to resubmit this article to another journal.



####  Robust and Label Efficient Bi-filtering Graph Convolutional Networks for Node Classification

---  **已邀请**。



####  A Hybrid ARIMA-WANN Approach to Model and Predict Vehicle Operating States 

密西西比 大学   **送审** 



#### FSS-GCN: A Graph Convolutional Networks with Fusion of Semantic and

Structure for Emotion Cause Analysis

-- 哈工大深圳  **送审  to blyucs**



####  Improving adversarial robustness of deep neural networks by using semantic information 

lina wang

--- 四川大学  送审 



####  **"A hybrid deep learning approach for gland segmentation in prostate histopathological images"** (1)

RINGS algorithem.    

明确训练数据集、验证数据集和测试数据集。似乎没有验证数据集，这可能导致过拟合或不拟合的问题。作者应该注意这个问题。

This article lacks methodological and theoretical innovation, much more work should be completed.
1. This article concentrate on application of typical deep learning method on gland segmentation in prostate histopathological. Lack of basic  theoretical innovation .
2. "state-of-the-art",  not "state-of-art" , as well as some english writing details should be polished.
3. I can't see any ablation study in the experiment. 
4. Computational time is achieved on what platform or device shuold be claimed. And the flops or MACs of the model is suggested to measured and compared, as well as the parameters or memory consumption.
5. It is suggested to research and cite more classic papers related to machine learning and deep learning. 
6. In the experimental comparison, it needs to be explained whether the data of the comparison method is from the original statement or the author's reproduction. 

--- 已reject 

Considering the above review opinions, we think that the current manuscript is not suitable for publication in this journal, so we suggest that it should be submitted to a more suitable journal 





#### **LGAttNet: Automatic Micro-expression Detection using Dual-Stream Local and Global Attentions**

 others：major 

he attention mechanism proposed in LGAttNet encourages the network to focus on particular facial regions of human and achieves better experimental results in micro-expressions classification than existing methods.
There are still some confusion remain.
\1.   What are the input size of SM(Sparse Module), since there three different inputs where the size of upper/lower face seems different from that of the whole image. And same puzzle with the vector addition before DM.
\2.   If I understand correctly, the SM is for the feature extraction, so is the convolution layers in FEM. I am a little confused why to separate the network like that. If the FEM only has the concatenation and sigmoid layers and the SM has two blocks where the output of the first block is fed to GAM, I think it is the same architecture with LGAttNet and it's more comprehensible for me. So, I'm curious about the reason why to arrange your network like that.
\3.   What is the specific multiplication operation in LAM and GAM? Matrix multiplication and element-wise multiplication are both feasible but lead to different mechanisms. 

mine：major





#### A Novel Framework for Detecting Social Bots with Deep Neural Networks and Active Learning  --- CHIN

other ：  Major Revision
1. I can infer the meaning of 'RGA' from the article, but I don't know the abbreviation of 'DABA'. It is suggested to be clear in the article.

2. In the abstract: 'after performance evaluation, the results show that DABD is more effective than the state of the art baselines', it is recommended to add accurate data description, for example, what is the accuracy of the method used in this paper?

3. Is there any mistake in the sentence of 'In our work, a total of 171 iteration are performed, 3420 users are manually labeled, and finally a classifier with the
Accuracy of 0.9801 is obtained. ' in the last paragraph of section 3.3.2. Is the data of 3420 users 'manually labeled'?

4. From Figure 4, we can't get a clear sense of the impact of hi+1 on hi. It is suggested to modify the diagram to make the network structure clearer. In addition, the expression of attention layer is not standardized. It is recommended to consult relevant materials and standardize the diagram.

5. In the DABA system, the data that is not manually labeled is labeled with the characteristics selected by the author as the evaluation criteria. I think that the system annotates more data with its own standard and increases the data set, which is equivalent to selecting appropriate candidate samples with its own standard. Therefore, whether the additional training samples and test samples are more in line with the judgment standard of the system. Then the better performance of the learning method on the larger data set is not convincing, because compared with SWLD-20K data set, the SWLD-300K data set only adds samples that meet the DABA system standard. Please give the real value of some samples in the data set of SWLD-300K.

mine：major +  cite 



Data collection 

1. The motivation for this article is meaningful and technically sound. But before the acception, I suggested much more work should be done or revision:
   1. The description of the process of the Sina Weibo's data collection approach, such as detection bypassing, proxy, etc. can weaken, for it has nothing to do with the scientific research and experimental demonstration.
   2. F-score would be more accurate if it is described as F1-score.
   3. In the model, the ablation effect of BiGRU Block, Resnet Block, and the Attention layer is suggested to be conducted.
   4. It is suggested to quote some recent research progress related to deep learning, such as:
       [1] Structural property-aware multilayer network embedding for latent factor analysis, Pattern Recognition, vol. 76, pp. 228-241, 2018. 
       [2] Multi-label image classification via feature/label co-projection, IEEE Transactions on Systems, Man and Cybernetics: Systems, DOI: 10.1109/TSMC.2020.2967071
   5. Most of the current comparative experiments are traditional machine learning methods. Are there any comparisons between deep learning or reinforcement learning methods from other papers? (As described in the relevant work, DQL) 




The motivation for this article is meaningful and technically sound. The ideas are innovative. The experimental ideas in this article are clear and the writing is methodical. But before the acception, I suggested much more work should be done or revision, as the evaluation submitted to the author. 





盾构机：  mine   reject   others： yuanhang



####  **The Partial Response Network: a neural network nomogram** 

 Reviewer #3: 1. Abstract should be improved to highlight the novelty.
\2. In Introduction, some newly published results are suggested to add, which is useful to improve the quality of work.
\3. I think it may be better if Section 3 is included in Section 4. 
\4. The structure is not very reasonable. I Section 2, the authors only mentioned some methods, but they failed to provide their own design. And for the whole paper, I can't clearly find out the methods design, so the motivation is not demonstrated.
\5. The output including format, font is not good to present.
6.The references should increase some newly published relevant papers.



Reviewer #4: The paper presents a data classification method based on artificial neural network. 
The proposed method aims at selecting the most important features and, thus, generating interpretable classification models. The method has also been compare with conventional Multilayer Perceptron, Sparse Additive Model, Random Forest, Support Vector Machine, and Gradient Boosting Machine. In my opinion, the paper requires a lot of significant corrections and cannot be accepted for publication in KNOSYS.

Major comments:

\1. In the Introduction, the Authors focus on methods that generate interpretable models. However, the description is not clear. A key question that may be asked is: What do the Authors mean by interpretability / explainability of the classification model? The ability to select features is not enough to make the model interpretable. Perhaps, the following paper will help the Authors to elaborate on this issue:

[1] Arrieta A.B. et al: Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. Information Fusion 58, 2020, pp. 82-115.

Moreover, some sentences are incomprehensible and require a detailed explanation, e.g.: 
1a) "The aim of this paper is to introduce a methodology to generate from the MLP, a
model that is intelligible in the sense that it is its own explanation",

1b) "It is widely accepted that the best explanation of a simple model is the model itself, as it faithfully represents its response globally, not just in the vicinity of a given approximation" - 

What do "it is its own explanation" and "the best explanation of a simple model is the model itself" mean?

\2. The mathematical description presented in Section 2 is incomprehensible. No descriptions of signs/variables used in the formulas. The method proposed in this section belongs to the group of the so-called black boxes. Their natural alternatives are rule-based systems (e.g. decision trees, fuzzy rule-based classifiers - FRBC), which belong to the gray-box-type approaches. In particular, FRBCs can not only classify data, but also explain the decision made. This is done with decision rules that are easy to interpret and understand. The learning techniques of FRBCs (e.g. based on genetic algorithms or multi-objective evolutionary algorithms) are also able to generate models with the most important features. Moreover, they generate models with accurate, compact, interpretable, and easy to understand rules. In this context, the method proposed in the paper should be compare with such techniques and potentially advantages should also be presented. 

\3. The application part of the paper is relatively week. Section 3 is not necessary. Benchmark data sets available from UCI Repository are very known for the community. Moreover, selected data sets (used in the paper) have relatively small number of samples. These sets could be treated as basics benchmarks only. Authors should consider more data sets - larger and much complex ones.

\4. The results are inelegantly presented. For instance, what do "Frequency" or "Contribution to logit" mean in the charts? No comments for Fig. 3b (interaction Att2 vs. Att4). Comment for Fig. 8 contains only one sentence (at the end of Section 4) and it explains nothing.

\5. In order to find out the significant differences between the results of the compared  algorithms, the comparative analysis should contain more tests of statistical significance (e.g. Fredman and Iman-Davenport test). Not too clear mention of one test (McNemar test) is not enough.

Minor comments:
\1. For the convenience of reviewers, figures should be included in the text with appropriate descriptions (near the references).





####  **Attention embedded mobile neural networks for crop disease identification** 

Dear Editor:
    This article concentrate on application of typical deep learning method on crop disease identification. Lack of basic  theoretical innovation. Therefore, this article is not suitable for publication on this journal.



Dear author：
    The motivation for this article is meaningful and technically sound. But concentrate on application of typical deep learning method on crop disease identification,  is not suitable for this journal. It is highly recommended to resubmit to other technical application journals. Thanks!





This paper spends a large amount of space to introduce the basis of  Convolutional Neural Networks , loss function and evaluation method, etc. These are obvious to most of the readers. The actual innovative content of this paper is not sufficient, and it lacks basic ablation experiments and analysis. Therefore, I believe that it is not suitable for publication in this journal. Much more work need to be done. Thanks！



Dear Author: 
   The motivation for this article is meaningful and technically sound. It concentrates on application of typical deep learning method on Recipe Suggestion and Generation.
This paper spends a large amount of space to introduce the basis of  Convolutional Neural Networks and Deep Learning. These are obvious to most of the readers. The actual innovative content of this paper is not sufficient, and it lacks basic ablation experiments and analysis. Therefore, I believe that it is not suitable for publication in this journal. It is highly recommended to resubmit to other technical application journals. Thanks！



Dear Author:
    It is  debatable that feature interactions and 1X1 convolution operations can be emphasized as innovative points.  And from the similarity check result, the originality of this article is worth considering again. Much more work need to be done, thanks.



### PlexNet: An Ensemble of Deep Neural Networks for Biometric Template Protection

印度三哥   感觉好像没有什么 。 

Dear Author:
In this article, most of the content was introducing the basic methods of deep learning.  Unfortunately, no obvious innovation points in deep learning model and theoretical analysis were founded.  Therefore, I believe that it is not suitable for publication in this journal.Thanks





### CAE-CNN: Predicting transcription factor binding site with convolutional autoencoder and convolutional neural network

成都信息工程大学， 算是纯应用文章





#### Federated Convolutional Neural Network with Knowledge Fusion for Rolling Bearing Fault Diagnosis

纯应用文章 ， 武汉理工大学  -- 拟reject





####  DRIN: Deep Recurrent Interaction Network for Click-Through Rate Prediction 

太原理工   有些判断不了优劣， 属于点击预测任务 。 重复率有些高



船舶分类

已reject 



#### Recipe Suggestion and Generation using Bi-directional LSTMs based Ensemble of Recurrent Neural Networks and Variational Autoencoders

应用文章



2020.08.26 

Fault detection in gears using fault samples enlarged by combination of numerical simulation and generative adversarial network

带确定是否送审--- wenzhou 



Adaboost-based reinforcement integration framework coupled two-stage feature extraction with deep learning for multivariate exchange rate prediction





DNN-DTIs: improved drug-target interactions prediction using XGBoost feature selection and deep neural network

药物靶向预测，    青岛科技大学。 





Astrologer: Exploiting Graph Neural Hawkes Process for Event Propagation Prediction with Spatio-temporal Characteristics

送审。。。 不懂





A Modified Convolutional Neural Network for Imbalanced Data Classification

拟reject 。 --- 台湾  损失函数的设计或者一些trick 。    

Dear Author:
    For this manuscript, we do not think that simple improvements of the loss functions and the application of the training tricks on the basic classification task are sufficiently innovative to be published on this journal. It is suggested that more work be added or resubmitted to other types of journals.





A Comprehensive Comparison of Meta-heuristics and Neural Network 
Approaches for Streamflow Time Series Forecasting 

拟 reject   过多的方法对比，介绍 ，? 神经网络的应用。 

Dear Author：
    After the preliminary reading and evaluation of the article, we found that a large amount of this paper is devoted to the introduction of various methods, and finally it is proposed that the method of deep neural network will result in better performance on Streamflow Time Series Forecasting task. We think that the innovation of this paper is not enough. Considering this, it is not suitable to be publish on KBS.  Thank you for your submitting. 



**Modeling of Cutting Parameters in Turning of PEEK Composite Using Artificial Neural Networks and Adaptive Neural Fuzzy Inference Systems**

应用文章， 拟reject 





 **Rule-based Reinforcement Learning Methodology to Inform Stochastic Algorithms for Constrained Optimization** 

--  强化学习， MIT 送审



 Unsupervised feature selection via transformed auto-encoder 

--- 福州大学， 拟送审



SMANN: A Segmented Memory-augmented Neural Network Using Sparsity Projection for Long-term Memorization

short communication  





Role-based Network Embedding via Structural Features Reconstruction with Degree-regularized Constraint



How to Select Deep Learning Networks for Image Classification

Dear Editor:
    This paper is devoted to the experimental comparison of popular deep neural network structures. It is more like a research review or experimental report, and we think its research value is ambiguous. At the same time, no innovative ideas were put forward. It is not suitable for our journal.



#### Development of a Lévy Flight and FDB-Based Coyote Optimization Algorithm for Global Optimization and Real-World ACOPF Problems

​    Please check it carefully that some pertinent suggestions have been put forward by reviewers, which is very helpful to improve the quality of your manuscript. In addition, we regret to inform you that we must terminate your submission process and suggest that you try another journal or conference. Thanks.  





2020.09.08

#### Chart Classification Based on Structural Variation Features Using Convolutional Neural Network

图表分类， 没有任何创新点。 仅应用。 





#### Forecasting Precious Metal Price using a Multiscale based Convolutional Neural Network Model

首先用VMD 模型从曲线中提取特征， 然后用CNN 来进行非线性建模， 预测价格。 



#### Intelligent Human Action Recognition Using an Ensemble Model of Evolving Deep Networks with Swarm-based Optimization

BI-LSTM   可以送审。 



#### STGSN - A Spatial-Temporal Graph Neural Network Framework for Time-Evolving Social Networks    --- 电子科技大学  qingke 

1. 犯罪网络分析， 社交网络分析，  图神经网络，  时空图网络，  注意力机制。 
2. 利用图网络对空间 （相互关系） 来进行建模， embedding   
3. 利用时空网络（多头注意力机制）来对不同时间进行建模 。

1. 针对网络模型和实验i结果， 结果缺少基本的理论分析和解释 。
2. 针对多头注意力机制和单头注意力机制效果基本一致的情况。 是否有原因分析 ？  
3. 是否考虑了针对消融实验的设计。 例如不采用注意力机制的
4. 在4.1 中，我没有找到关于metric 的描述。 



#### Hierarchical Recurrent Neural Networks for Graph Generation

---- 送审lyubo



A big data system for driving performance evaluation system with unsupervised
 learning from bi-level distribution mixture of interpretable features

--- reject  应用型



 Dilated Convolutional High-Resolution Network for Crowd Countin  



desk reject 



Dear editor: 
   I’ve found that this article's similarity check rate reaches to 39%,  and it has a high degree of similarity to another published article. I suggest the desk reject.

Dear author: 
   I’ve found that this article's similarity check rate reaches to 39%,  and it has a high degree of similarity to another published article.  Please carefully check your writing details.



​    I’ve found that this article's similarity check rate reaches to 35%,  and my preliminary review reveals some duplicate paragraphs.  Please carefully check your writing details.



####  Information Aggregation and Fusion in Deep Neural Networks for Object Interaction Exploration for Semantic Segmentation 

---   北京交通大学   同已reject 的文章 摘要相似度高， 粗略看是语义分割的， 貌似创新一般。 拟reject 。 

#### Dilated Convolutional High-Resolution Network for Crowd Counting

--- reject 

#### Fast Modular Assembler Encoding: Evolving Modular Neural Networks of fixed modular architecture

-- 送审 

####  High-resolution convolutional neural networks for point clouds with space decomposition kernel 

---- HRNET+ SDK conv 



#### Cyberbullying Classification with Neural Networks

Dear author：
      This paper is devoted to the Cyberbullying Classification with Neural Networks. The topic do make sense. But it is more like a research review or experimental report, and we think its research value is ambiguous. At the same time, no innovative ideas were put forward. It is not suitable for our journal.  Sorry to suggest the reject.



####  Representation learning using Attention Network and CNN for Heterogeneous networks 

-- 浙江工业大学，  拟送审 ，GCN  +  attention



#### Differences First in Asymmetric Brain: A Bi-hemisphere Discrepancy Convolutional Neural Network for EEG Emotion Recognition

--- reject 

#### Research on Imbalanced Data Classification Based on Classroom-like Generative Adversarial Networks

-- 送审 



#### Deep Learning Based Extractive Text summarization with Word-Level Attention Mechanism

-- reject 。



#### A synchronous prediction model based on multi-channel CNN with moving window for coal and electricity consumption in cement calcining process

--- 创新一般。 



#### Fusion of convolution neural network, support vector machine and Sobel filter for accurate detection of COVID-19 patients using X-ray images





 This work focuses on many-objective optimisation problems and proposes a co-guided evolutionary algorithm by combining merits of dominance and decomposition, in which an elitism mechanism based on the cascading sort is exploited to balance the convergence and diversity of evolutionary process. Also, a reference point adaptation method is designed to adapt different Pareto Fronts. This is quite an interesting and valuable topic in related areas. However, some aspects need to be improved.

My detailed comments are as follows:
\1. On the title:
As mentioned in the statement of main contributions, this study utilizes both dominance-based and decomposition-based approaches to balance the convergence and diversity of the evolutionary process. However, only one is included in the title, as A Dominance-based Many-objective Evolutionary Algorithm With Reference Point Adaptation. In my opinion, it cloud be better to get most of your contributions revealed by the title.
\2. On the Motivation:
As a reader, the current Motivation is not that satisfactory. Please revise this part to convince readers that all your contributions/improvements are necessary and reasonable.
\3. On the Experiment:
1) The selection criterion of algorithms to compare is missing. Moreover, no algorithm later than 2018 gets selected. But, some algorithms released within two years, which means those are newer, get covered by the authors' literature review, not to mention others that can be easily found from well-known journals. Please add more discussion on this.
2) Detailed experimental studies are expected, including (1) the parameter study since authors claimed that "The proposed algorithm does not contain sensitive parameters and has promising versatility in MaOPs optimization with various PF" in the 3rd contribution; (2) experiments that can prove the effectiveness of each contribution, making the study complete and also comprehensive.
\3. Grammatical mistakes and typos:
Please carefully conduct proofreading and correct grammatical mistakes and typos, e.g., section title "Detailed implementations", given that "implementation" is a mass noun. 



AE: This article is a re-submission as a new paper, but the duplication check result with the former rejected paper is of 95%,  for the KBS rules “any rejected papers must make >50% changes for re-submission as a new paper.” I suggest the desk reject.

AE: This article is a re-submission as a new paper, but the duplication check result with the former rejected paper is of 91%. For the submission rules of KBS, I suggest the desk reject.

AE: This article is a re-submission as a new paper, but the duplication check result of abstact with the former rejected paper is of 95%.In addition, From my pre-review of this article and the former submission,  I found no enough revise.  I suggest the desk reject. 