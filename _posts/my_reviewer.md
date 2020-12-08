#### A Novel Framework for Detecting Social Bots with Deep Neural Networks and Active Learning

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





#### Constraint Interpretable Double Parallel Neural Network and Its Applications in the Petroleum Industry





#### Forecasting Precious Metal Price using a Multiscale based Convolutional Neural Network Model

Dear author:
   This article provides a new way of thinking in the field of financial analysis, which is instructive for readers and future research work, but i believe there exist some problems in terms of the writing and experiment:
    1. For the professional reader of the journals e.g. KBS,  so much contents with respect to the basic instrctuion of CNN is not proper. 
        2. Besides, I can't found the comparsion methods are random walk and ARMA model, which are foundamental models. 
        3. No ablation study is presented, how about the VMD combined with other regression models, or provide the sufficient reason and theoretical analysis to choose the VMD. 





表述：

1. RNN 本来即可是多层的，在我看来，本文没有强烈的意义去强调Hierarchical，
2. 未对文章意义进行详细的阐述。这个任务完成后的意义？ 图结构的生成，实现数据增强？或者进行未知图结构的设计？
3. 为什么又有SGD, 又有ADAM ？？？
4. We adopt two metrics to evaluate ...: 6) the distribution of the node
labels, 7) the distribution of the edge labels and 8) joint distribution of node labels and 235
degree, which the eighth takes into account both structure and labels. ----- There exist some logic errors, present two or three. These kinds of  writing error are also found somewhere else.

创新性：
1. 这个课题具有一定创新性 
2. 在GraphRNN这篇工作，同样也是利用RRN进行图网络的生成来说，该文章的创新程度一般。

实验：
1. 没有同其他的生成模型GAN,VAE 做对比
2. 没有实验超参，实验细节，数据集划分。消融实验
3. 对比方法来源于复现or 其他文章申明？？？
4. 未对实验结果进行有效的分析，仅仅时呈现，略显生硬
结论：
1. 结论weak ，未对工作进行正式的总结陈述，以及对未来的展望

Comments for writing and representation:
1. RNN could be multilayered originally. In my opinion, this paper has no strong meaning to emphasize Hierarchical,
2. The significance of the article is not elaborated.What is the significance of this task?Graph structure generation, data enhancement?Or the design of unknown graph structure?
3. As for the optimizer,  MBGD and ADAM?  “we adopt MBGD which combines the advantages of both BGD and SGD. We select Adam [39] as our optimizer,”
4. “We adopt two metrics to evaluate ...: 6) the distribution of the node
labels, 7) the distribution of the edge labels and 8) joint distribution of node labels and 235
degree, which the eighth takes into account both structure and labels”. ----- There exist some logic errors, present two or three. These kinds of  writing error are also found somewhere else.

Comments for innovative:
1. This subject of this article is innovative enough.
2. In base of the privious work, GraphRNN, in which an RRN was also utilized to generate graph network, it seems that the innovation of this article is limited.


Comments for experiment:
1. No comparison was made with other generated models GAN and VAE
2. No experimental hyperparameter, experimental seeting and dataset split are provided. The ablation experiment is lack.
3. The comparison method is derived from reproducing or other articles stating?? It should be specilized.
4. There was no effective analysis of the experimental results, which were only presented straightforwardly.

Comments for conclusion:
Weak, it does not formally summarize the work, let alone the formally  looking forward to the future.


【新智元导读】图网络领域的大牛、斯坦福大学Jure Leskovec教授在ICLR 2019就图深度生成模型做了演讲，阐述了图生成模型的方法和应用，并详细介绍了他的最新成果

斯坦福大学教授Jure Leskovec是图网络领域的专家，图表示学习方法 node2vec 和 GraphSAGE 作者之一。

图神经网络将成 AI 下一拐点！MIT 斯坦福一文综述 GNN 到底有多强

此外，在 ICLR 受邀演讲上，Jure Leskovec 教授还就图深度生成模型做了演讲。在这次演讲中，Jure 阐述了图生成模型的方法和应用，并详细介绍了他的最新成果，GraphRNN 和 Graph Convolutional Policy Network。



#### Frame-based Neural Network for Machine Reading Comprehension

This article propose novel attention-based Frame Representation Models, a new Frame-based Sentence Representation (FSR) model, Frame-based Neural Network for MRC (FNN-MRC)
Base on these models, the experimental results based on Frame-based Neural Network for MRC(FNN-MRC) demonstrates significant results on MRC task. This paper is well-organized, an innovative framework to tackle the MRC problems is proposed.
In my opinion, this paper worth to be spread widely to the readers in this area. 
Some minor problems need to be considered carefully:

1. The figures in the paper are not displayed in a professional way, e.g. figure 2, and figure 3.
2. The writing details need to be polished, native English writing is suggested.
3. Much more notation explanation about the Equation, e.g. Eq.10 Eq.11 are prefered.



