#### meta-learning   (learning to learn)

对meta-learnig 的介绍 [meta-learning](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)
Model-agnostic meta-learning for fast adaptation of deep networks

### So, what is learning to learn, and what has it been used for?

Early approaches to meta-learning date back to the late 1980s and early 1990s, including [Jürgen Schmidhuber’s thesis](http://people.idsia.ch/~juergen/diploma.html) and [work by Yoshua and Samy Bengio](http://bengio.abracadoudou.com/publications/pdf/bengio_1991_ijcnn.pdf). Recently meta-learning has become a hot topic, with a flurry of recent papers, most commonly using the technique for [hyperparameter](https://arxiv.org/abs/1502.03492) and [neural](https://arxiv.org/abs/1703.00441) [network](https://arxiv.org/abs/1703.04813) [optimization](http://www.cantab.net/users/yutian.chen/Publications/ChenEtAl_ICML17_L2L.pdf), finding [good](https://arxiv.org/abs/1611.01578) [network](https://arxiv.org/abs/1611.02167) [architectures](https://arxiv.org/abs/1704.08792), [few](https://arxiv.org/abs/1606.04080)-[shot](https://openreview.net/forum?id=rJY0-Kcll) [image](https://arxiv.org/abs/1703.03400) [recognition](https://arxiv.org/abs/1606.02819), and [fast](https://arxiv.org/abs/1611.02779) [reinforcement](https://arxiv.org/abs/1611.05763) [learning](https://arxiv.org/abs/1703.03400).
*Various recent meta-learning approaches.*

![](D:\00_code\blyucs.github.io\images\meta-learning\banner.jpg)





# How Recent Meta-learning Approaches Work

Meta-learning systems are trained by being exposed to a large number of tasks and are then tested in their ability to learn new tasks; an example of a task might be classifying a new image within 5 possible classes, given one example of each class, or learning to efficiently navigate a new maze with only one traversal through the maze. This differs from many standard machine learning techniques, which involve training on a single task and testing on held-out examples from that task.

![](D:\00_code\blyucs.github.io\images\meta-learning\meta_example.png)

 *Example meta-learning set-up for few-shot image classification, visual adapted from [Ravi & Larochelle ‘17](https://openreview.net/forum?id=rJY0-Kcll).* 

 During meta-learning, the model is trained to learn tasks in the meta-training set. There are two optimizations at play – **the learner, which learns new tasks**, and **the meta-learner, which trains the learner**. Methods for meta-learning have typically fallen into one of three categories: recurrent models, metric learning, and learning optimizers. 



MAML  预训练 差别， 预训练是在当前预训练数据集上算loss ， 而MAML 是在测试任务上评估算loss

![](D:\00_code\blyucs.github.io\images\meta-learning\MAML.png)



#### Omniglot and miniImagenet  

![](D:\00_code\blyucs.github.io\images\meta-learning\omniglot.png)

#### support set and  test (learn and test by learner)

![](D:\00_code\blyucs.github.io\images\meta-learning\support_and_test_0.png)

![](D:\00_code\blyucs.github.io\images\meta-learning\support_and_test_1.png)

Omniglot有时被成为mnist的转置，因为它有1623类字符，每类只有20个样本，相比mnist的10个类别，每个类别都有上千样本，正好相反。 



##### training set 

##### sample set / query set    ---- meta learner

（CxK    C-way,K-shot , sampled from training set） 

This sample/query set split is designed to simulate the support/test set that will be encountered at test time. A model trained from sample/query set can be further fine-tuned using the support set, if desired. In this work we adopt such an episode-based training strategy. 

##### support set / test set  ---  learner



在迁移学习中，由于传统深度学习的学习能力弱，往往需要海量数据和反复训练才能修得泛化神功 。为了 “多快好省” 地通往炼丹之路，炼丹师们开始研究 Zero-shot Learning / One-shot Learning / Few-shot Learning。

爱上一匹野马 (泛化能力)，可我的家里没有草原 (海量数据) 。

- Learning 类型分为：Zero-shot Learning、One-shot Learning、Few-shot Learning、传统 Learning。



## Zero-shot Learning

Zero-shot Learning，零次学习。

成品模型对于训练集中没有出现过的类别，能自动创造出相应的映射：![X \rightarrow Y](https://math.jianshu.com/math?formula=X%20%5Crightarrow%20Y)

既要马儿跑，还不让马儿吃草。

就像人的“触类旁通”。

举个简单的例子。

假设1，小明知道斑马是一个黑白条纹，外形像马动物。

假设2，小明未见过斑马。

小明见到斑马大概率能认出这个是斑马。

零样本学习就是这种：可见类学习模型+辅助性知识、边缘描述+判断识别的过程。

## One-shot Learning

One-shot Learning，一次学习。

训练集中，每个类别都有样本，但都只是少量样本。

既要马儿跑，还不让马儿多吃草。



## Few-shot Learning

Few-shot Learning，少量学习。

也即 One-shot Learning 。



## 传统 Learning

即传统深度学习的**海量数据** + **反复训练**（炼丹模式）。

家里一座大草原，马儿马儿你随便吃。



AUTOML

一整个pipeline 的自动化， 具有很强的工业化应该背景。

NAS 的意义 

强化学习的优势， 劣势 

连续空间算法的优势，劣势



#### SMASH: One-Shot Model Architecture Search through HyperNetworks 

超网络，内存读写， bank    SMASH通过学习一个辅助超网来近似模型权重，从而绕过了对候选模型进行完全训练的需求，从而能够以单次训练运行为代价，快速比较大范围的网络架构。 

We propose a technique to accelerate architecture selection by learning an auxiliary **HyperNet** that generates the weights of a main model conditioned on that model’s architecture. By comparing the relative validation performance of networks **with HyperNet-generated weights**, we can effectively search over a wide range of architectures at the cost of a single training run.

![](D:\00_code\blyucs.github.io\images\meta-learning\memory-bank.png)



**one-shot 体现在哪里** ？  trained once ，weight-sharing

(1) Design a search space that allows us to represent a wide variety of architectures using a single one-shot model. 

(2) Train the one-shot model to make it predictive of the validation accuracies of the architectures. 

(3) Evaluate candidate architectures on the validation set using the pre-trained one shot model. 

(4) Re-train the most promising architectures from scratch and evaluate their performance on the test set.



#### Understanding and Simplifying One-Shot Architecture Search

对SMASH的one-shot 方法的**解读，论证， 简化**

作者比较了one-shot模型的精度和stand-alone模型的精度，发现两者之间确实存在强相关性，这也证明了one-shot方法的合理性。  引入choice block 机制简化。 

![](D:\00_code\blyucs.github.io\images\meta-learning\understanding_error.png)

####  《DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH》

#### 《PROXYLESSNAS: DIRECT NEURAL ARCHITECTURESEARCH ON TARGET TASK AND HARDWARE》

#### 《Single Path One-Shot Neural Architecture Search with Uniform Sampling》

#### 《Once For All: Train One Network And Specialize It For Efficient Deployment》 



#### 《SqueezeNAS: Fast neural architecture search for faster semantic segmentation》   ---  DeepScale Inc.

质疑了三种假设： 

1.  The most accurate neural network for ImageNet image classification will also be the most accurate backbone for the target task    --- -directly on the target task
2. Neural Architecture Search (NAS) is prohibitively expensive  ---  Use modern supernetwork-based NAS
3. Fewer multiply-accumulate (MAC) operations will yield lower latency on a target computing platform.  --  optimize for both accuracy (on the target task) and latency (on the target platform)  ---platfrom-awared

This search space was chosen to be similar  to the FBNet[9], MobileNetV2[40], and MobileNetV3[36]

使用cityscapes 街景数据集   ，    没有开源训练过程，只有网络结构

![](D:\00_code\blyucs.github.io\images\meta-learning\squeezenas.png)



《Scaling Up Neural Architectures Search With Big Single-Stage Models》

《Searching for A Robust Neural Architecture in Four GPU Hours》

《Single Path One-Shot Neural Architecture Search with Uniform Sampling》

《One-Shot Neural Architecture Search via Self-Evaluated Template Network》

《MnasNet: Platform-Aware Neural Architecture Search for Mobile》

《Fbnet:Hardware-aware efficient convnet design via differentiable neural architecture search》

one-shot + DARTs 



#### NAS 过程

1. 限定整体结构  （supernet/ one-shot / cell-based / block-based /  可微结构+ cell stack/ backbone / encoder-decoder ）
2. 待搜索的连接/结构  (cell 之间的连接 ，cell 内部的连接， 随机从supernet 中选取候选架构)
3. 搜索空间 （operations  ： residual ，ASPP，atrous， dep-wise， point-wise,  group -conv）
4. 搜索策略（基于hypernet 的随机搜索，可微搜索，强化学习搜索）
5. 搜索加速（使用代理任务在迁移到目标任务 （例如cifar10 上搜索，迁移到imagenet上训练） ， 或者是直接基于target task  ， early-stopping ， pre-determined）
6. 是否platform-aware / resource-constrainted / multi-objective / 目标函数加入正则化项   （latency，MACs， FLOPs）
7. 合理性讨论 -  reward predicts the final score , one-shot   strong correlation  with the  stand-alone





Q？ 

face recognition 如果新加一个人进来， 是否需要重新训练 ？  使用verification 来解决recognition 问题 。  不是使用CNN+SOFTMAX 的分类架构。

每个人的照片只有一张，那么模型是如何得到训练的 ？  







NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection












