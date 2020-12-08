0801本周工作：

1. pre-trained ， KD and   分阶段。--- 准备工作。
2. 训练时间的问题    ---  **结合pre-trained KD**， **结合划分阶段训练**， 调整训练epoch ， batchsize， LR 等超参， 过程优化。 **从之前的8分钟优化到约70s**，一天可以跑1000轮，基本可以达到收敛。
3. 模型鉴别度的问题。 1）解决出现两头多的问题 。 通过更加准确的f1-score  （不采用overall 的eyes，brows，而是单独的）给到reward，**效果明显** 。   2）修正了数据增强里的**random mirror** 导致的左右属性错乱问题（**针对让人脸**）。
4. 参数量分层问题  --- 架构单一导致。调整了整体的架构超参（cell-num and branch-num）， 
5. 自动化数据分析 --- 提升分析效率
6. 解决诸多隐藏的代码bug 。
7. 审稿相关 

下周计划：
1. 审稿相关
2. 主要multi-objective 相关实验，**消融实验**基本跑完，性能数据摸底出来。 
   





0808本周工作：

1. 基于params 的搜索 消融
2. 基于flops 的搜索 消融， 新增 drop 方案 ， 不对过小架构做奖励
3. 基于delay 的搜索 消融，  尚在进行中。 
4. 数据分析，整理， 可视化，pareto 图等代码编写 
5. 审稿相关

下周计划：

1. 实验，数据整理， 论文撰写
2. 审稿相关



0912 本周工作：

1. PROXYLESSNAS: DIRECT NEURAL ARCHITECTURE SEARCH ON TARGET TASK AND HARDWARE

   **论文的初衷 **  --- 

   1）time  / memory ： 为了省时间用gradient ， 但是带来显存增加。 

   

   ![image-20200912143601400](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912143601400.png)

    分别实现了可微方案和强化学习两种方案。 

   用hansong 论文提出的模型，在imagenet 100 上训练300 epochs， 结果如下： test-acc1  =  79 %

   ![image-20200911094027697](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200911094027697.png)

2. HAQ   HAQ: Hardware-Aware Automated Quantization with Mixed Precision

   **论文的初衷**:DDPG (actor+critic )**逐层的混合精度量化**， 某一层的冗余情况是不同的，需要的精度是不同的， 因此量化效果也不同。 

   ![image-20200912145845178](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145845178.png)

   可以做的事情：

   由于该文章使用的是accelerater simulator（BISMO） ， 针对特定架构的做好了一个lookup table (mobilenetv2)， 输入（层数，weight 位宽， activation 位宽）， 输出。 不涉及硬件平台的直接反馈。 

3. MnasNet: Platform-Aware Neural Architecture Search for Mobile

![image-20200912145640806](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145640806.png)



**基于一个前提：**

1. **不同的平台需要不同的架构（NAS 需要 定制化specific）**

   ![image-20200912144905040](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912144905040.png)

   ![image-20200912145748244](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145748244.png)

2. **FLOPs 不能直接作为效率指标**

​    ![image-20200912145041421](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145041421.png)

基于这个前提，可以做的事情是：

1. **基于单板的platform-aware proxylessnas**  : 工程开销，模型传输，时间开销，自动化pipeline 的建立。 创新程度不够。 （以proxylessnas-R 为基线）   

2. **基于单板的混合精度量化搜索 。 困难： 量化框架 （ncnn?????   8bit 16 bit  32bit ） **（以HAQ: Hardware-Aware Automated Quantization with Mixed Precision为基线）--- 创新一般， 工程意义很大。 

3. **Offline multi-objective platrform-aware NAS  on target device** (**可以以 RL-based NAS segmenter 为基线**) ---  困难：结构超参要做 编码word2vec （输入特征 ？ ？？？）--- 当前不需要做针对所有类型的网络，在此只关注我们针对NAS 所设计的网络，这样结构超参具有统一特定的语义。 有点像NLP 问题， LSTM 。  --- 创新较大  --- **当前没有人做过**（有针对分布式系统的时延估计，Artificial Intelligence-Based Latency Estimation for Distributed Systems、 The Correct Way to Measure Inference Time of Deep Neural Networks） --  和强化学习方法过程天然契合，现成的结构编码，训练/采样/评估独立， 也是可微方式（DARTs）所不具备的。 

   ![image-20200912150127007](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912150127007.png)

   ![image-20200912150038977](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912150038977.png)

4. multi-objective platform-aware NAS    (**以 RL-based NAS segmenter 为基线**)   ---- 困难：环境/移植/板间通信 等工程开发工作量。 --- 创新很一般。 

| 工作                                                         | 困难                                                         | 创新性 | 工程意义 | 工作量 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------ | -------- | ------ |
| **基于单板的platform-aware proxylessnas**                    | 工程开销（环境/移植/板间通信 等工程开发工作量。），模型传输，时间开销，自动化pipeline 的建立。 | 一般   | 还行     | 小     |
| 基于单板的混合精度量化搜索                                   | 量化框架 （ncnn?????   8bit 16 bit  32bit ）                 | 一般   | 大       | 巨大   |
| **Offline multi-objective platrform-aware NAS  on target device** | latency估计模型的建立及其准确度。                            | 较大   | 大       | 大     |
| multi-objective platform-aware NAS（基于nas segmenter）      | 环境/移植/板间通信 等工程开发工作量。                        | 低     | 还行     | 小     |
| 跨单板的**分布式**推理时延                                   | 框架（在Flink上使用Analytics Zoo进行实时、分布式深度学习模型推理） |        | 大       |        |
| self-adaptive multi-objective NAS                            | 对multi-objective 超参的调整  自动化                         | 还行   | 大       | 大     |
| 一种通用的时延估计方法（多个典型的搜索空间）                 | 不同的运行平台，不同的架构模式                               |        |          |        |
| 一种通用的精度估计方法（多个典型的搜索空间）                 | 不同的架构模式， 走多目标平台，离线化，无需部署的路子。      |        |          |        |
| 强化学习+dense per pixel 问题， accuracy prediction辅助提升效率。 | 以提升效率为目的，以fast seg 为蓝本。 走快速选择好的网络，误需训练 searched 网络的路子。 |        |          | FLOPs  |
| 隐约感觉图网络的方法可以实现latency prediction 的不同架构模型（darts，nasnet，mobilenet，）下的大一统 |                                                              |        |          |        |
| autodeeplab+latency 做。 ？？？                              |                                                              |        |          |        |
| OFA的mobilenetV3 预训练模型 +加一个语义分割头+               |                                                              |        |          |        |
| performance 和latency 之间是有关系的， 两个监督信号，相互补充。双分支，互为监督。 |                                                              |        |          |        |
|                                                              |                                                              |        |          |        |





latency **估计**

![image-20200912145445343](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145445343.png)

![image-20200912145501188](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145501188.png)

![image-20200912145516917](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145516917.png)





**2020.10.16：** 

本周工作： 

1. once-for-all 论文。

2. 环境配置， 分布式训练环境（horovod）。

3. 相关引申论文阅读。

4. 审稿相关

   

#### once-for-all 

为什么要做once-for-all？

###### We need GREEN AI



![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017130816.png)

###### Diverse platforms and scenerios, NAS is expensive. We need Once-for-all:



![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017131103.png)

![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017124612.png)

如何做的？ 

##### 一、Train一个OFA network （Train） 

##### （从模型压缩的思路入手，从大的开始，逐渐train 小的，同时又兼顾大模型。 避免模型之间的相互干扰。最终达到一个效果是： 小模型，中模型，大模型都可以拿来即用。）   

##### ![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017124359.png)

1. 渐近收缩 （**progressive shrinking**）

   ![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017124954.png)
2. 矩阵转换（将大卷积核转换为小卷积核）

   ![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017125408.png)

   ##### 二、从OFA network产生 platform-specilized sub-network （Search）

   ###### 1. 精度预测， 时延预测 

We use **a three-layer feedforward neural network that has 400 hidden units** in each layer as the
accuracy predictor. Given a model, we encode each layer in the neural network into a one-hot vector
based on its kernel size and expand ratio, and we assign zero vectors to layers that are skipped.

![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017125719.png)

2. ###### evolution search 



1. traditional  :  search -- train --search --train -
2. proxylessnas:  train+ search  ---  train + search

3. OFA:    train --> search . 先train 再search , without any finetune. 

效果：

1. 精度时延预测的效果

2. once-for-all 的efficient效果（by search ）

   ![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017144513.png)

   

   ![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017124441.png)

3. 模型精度

   ![image-20201017124817422](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201017124817422.png)



Efficient Architecture Search by Network Transformation  --  cai han



#### ACCELERATING NEURAL ARCHITECTURE SEARCH USING PERFORMANCE PREDICTION

**---2017 年 --利用 支持向量机回归模型(ν-support vector machine regressions(ν-SVR).)    ν-SVR with RBF kernels**

1. 提出在超参调优，元建模方法， NAS 方法中， accuracy prediction significantly improving the efficiency.

   例如 early stopping ， reward 反馈 

![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017095158.png)

 

![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017101417.png)

Learning curve prediction with bayesian neural networks



#### One- shot  NAS 

Single Path One-Shot Neural Architecture Search with Uniform Sampling  ---旷视

Understanding and Simplifying One-Shot Architecture Search

Searching for A Robust Neural Architecture in Four GPU Hours  -- yangyi 

One-Shot Neural Architecture Search via Self-Evaluated Template Network  ---yangyi

#### 下周计划

1. 详细研究accuracy prediction 和 latency prediction 代码
2. 详细研究 evolution search 部分代码







本周工作：

1. **accuracy prediction** 部分， **latency prediction** 部分代码。 学习了如何使用预训练的OFA。 运用进化搜索算法，基于一个OFA的超网络，搜索出了特定latency constraint 下的model 。  

   ![](D:\00_code\blyucs.github.io\images\weekly-report\QQ截图20201017125719.png)

   1. Learning curve prediction with bayesian neural networks

   2. ACCELERATING NEURAL ARCHITECTURE SEARCH USING PERFORMANCE PREDICTION

   ![image-20201024142148481](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201024142148481.png)

   

2. HANSONG 等人做了  latency prediciton ，但还是查表、累加的方式，并且只公布了三星NOTE10上了lookup table ，受限于note10 的部署条件，我们也无法验证latency是否准确。

   “1080TI，intel CPU，JETSON， google pixel 1” 等OFA 论文所提出支持的设备，全部没有公开lookup table。

    **所以我们来尝试研究这个问题。**  类NLP 问题。

3. 整理研究思路， 研究计划。 OFA 我们训练不起， 那么直接把OFA的代码当成一个基于MOBILENET V3搜索空格键的生成器来用。不适用OFA的中weights， 仅涉及架构选择。

4. 编写代码， 基于mobilenetV3 搜索空间，采集基于GPU/ CPU  的数据集 20000 条。 复用现成的encoding 方式，快速完成实验。当前正在建模拟合。   

5. GCN/GNN 基础理论。 



下周计划

1. Latency prediction 实验。
2. 图网络学习  基础论文的代码研究 。 





强化学习+dense per pixel 问题， accuracy prediction辅助提升效率。  以提升效率为目的，以fast seg 为蓝本。 走快速选择好的网络，误需训练 searched 网络的路子。

1. RL-base 的好处是相对于连续空间，超网络，共享参数一脉的NAS， 可以探究更加广的搜索空间，不易陷入局部最优。
2. 其缺点是，每一个候选架构需要独立训练，这个是非常耗时的，尤其是在dense-per-pixel 这类任务下， 
3. 因此我们能否，接用精度预测？？？那么数据集从何而来， 还是必须得训练特别多得模型，然后采数据集。-------这个故事讲不下去了。。。。。。。























2020.10.31：

1. **Predictor-Based Search Methods**

   1) **Latency  prediction**  :   

   ​         **现状**：近一年，约3篇研究，基于 LUT accumulate 的方式； **end-to-end prediction 的仅一篇**；根本问题是缺少可用数据集。

   ​        **计划：**MobileNetV3 的搜索空间上采集latency数据集，基于粗略的“全连接网络” 实现end-to-end 的latency prediction。   RMSE  3ms。  a) 修改编码方式     c) 样本均衡化处理   b) 模型

   2) **Accuracy prediction**： 

   ​		**现状：**近一年, 约10 篇左右文章涉及该问题，研究基于相关数据集的精度预测， 用于改进NAS方法。

   ​        **计划：**重点研究 NAS-Bench-101 ， NAS-Bench-201 ， NAS-Bench-301  三个数据集论文和代码。以及 “ Encodings for Neural Architecture Search ” 论文。 

2.  **GNN / GCN** 

   1)GCN 代码运行，理解 及公式推导。 基于cora （论文引用关系）数据集

   2)GAT 代码运行，理解。   基于cora 数据集

   3) GNN 基础

3. **BRP-NAS: Prediction-based NAS using GCNs**

   1）CNN --> encoding-->GCN--> latency/accuracy .

   2）数据集：NAS-Bench-101,  NAS-Bench-201 DARTS.  LatBench（未公开）

   3)  宣称公开dataset and code  ， 尚未公开。



2020.11.7：

1. **Latency  prediction**  :  

   <img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201107145751183.png" alt="image-20201107145751183" style="zoom:33%;" />

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201107145825960.png" alt="image-20201107145825960" style="zoom:33%;" />

2.  **Accuracy prediction**： NAS-Bench-101 ， NAS-Bench-201 ， NAS-Bench-301  三个数据集论文和代码

3.  论文修改





2020.11.14：

1. latency prediction ： 完成了基于1080TI部署加速引擎TensorRT，具备“可信、合理”数据集采集的条件，实现了模型约 10~ 20 倍的加速 。

   下周计划： 数据集部分工作 ； JETSON NANO 版上部署加速引擎TensorRT的配置；模型优化。

2. 论文修改，实验

   1）环境原因造成模型丢失，恢复训练

   2）trade-off 的模型训练（基于HELEN, EG1800）

   3） UNET + celebA的训练

   4）deeplabV3 + mobileNetV2 训练 （基于HELEN, EG1800）

   下周计划：  

   1）deeplabV3 + mobileNetV2 训练

   2）其他NAS 方法适配数据集，搜索，训练

   3）审稿相关



2020.11.21：****

**本周工作：**

1. 审稿相关  GRAPHRNN

2. latecncy predictor：基于1080TI + TensorRT引擎采集数据集 ， 当前完成大约 1.5W+，由于需要做模型转换，速度很慢.  除了 LSTM ， 还调试了GRU，GBDT，LightGBM 模型。 当前决策树的生成还在调试中。

3. 论文修改， response letter

**下周计划：**

1. 博士生综合考试

2. 实验补充

3. 论文修改

   

2020.11.28：

本周工作

下周计划：

1. 论文修改完成





2020.12.5：

本周工作：

1. 论文修改， response letter 

下周计划：

1.  论文开题资料
2. 时延预测，停了两周。
3. TNSE 论文











