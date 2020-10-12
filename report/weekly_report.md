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
| 跨单板的分布式推理时延                                       | 框架（在Flink上使用Analytics Zoo进行实时、分布式深度学习模型推理） |        | 大       |        |
| self-adaptive multi-objective NAS                            | 对multi-objective 超参的调整  自动化                         | 还行   | 大       | 大     |

FLOPs



latency **估计**

![image-20200912145445343](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145445343.png)

![image-20200912145501188](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145501188.png)

![image-20200912145516917](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200912145516917.png)