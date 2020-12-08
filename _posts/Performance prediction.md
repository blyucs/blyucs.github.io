#### NPENAS: Neural Predictor Guided Evolution for Neural Architecture Search

西电的文章   ---  基本就是做了 OFA 的事情， 进化算法+ predictor  -- **accuracy** 

NPENAS-BO achieving state-of-the-art performance on NASBench-201 and  NPENAS-NP on NASBench-101 and DARTS, respectively.



--- 我们可以做一个： Performance-prediction based platform-aware Neural architecture search 



#### BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search ----  accuracy 

在流行的搜索空间中，我们只需使用**200个训练点**就可以将新架构的验证精度预测到其真实值的1%以内。

This may be of independent interest beyond NAS.

code：  https://www.github.com/naszilla/bananas

There are several works which predict the validation accuracy of neural networks (Deng et al., 2017;Istrate et al., 2019; Zhang et al., 2018), or the curve of validation accuracy with respect to training time (Klein et al.,2017; Domhan et al., 2015; Baker et al., 2017). A recent algorithm, AlphaX, uses a meta neural network to perform NAS (Wang et al., 2018). The search is progressive, and each iteration makes a small change to the current neural network, rather than choosing a completely new neural network.
A few recent papers use graph neural networks (GNNs) to encode neural architectures in NAS (Shi et al., 2019; Zhang t al., 2019). Unlike the path encoding, these algorithms require re-training a GNN for each new dataset. 



#### Bridging the Gap between Sample-based and One-shot Neural Architecture Search with BONAS  -- accuracy

 In this work, we propose
BONAS (Bayesian Optimized Neural Architecture Search), a sample-based NAS
framework which is accelerated using weight-sharing to evaluate multiple related
architectures simultaneously. Specifically, we apply a **Graph Convolutional Network predictor** as surrogate model for Bayesian Optimization to select multiple
related candidate models in each iteration.

This approach not only **accelerates** the traditional sample-based approach significantly, but also keeps its reliability.





#### LETI: LATENCY ESTIMATION TOOL AND INVESTIGATION OF NEURAL NETWORKS INFERENCE ON MOBILE GPU  --- latency 

 斯科尔科沃科技学院   --- 感觉很一般。 提供的GitHub库是空的。

We experimentally demonstrate the applicability of such an
approach on a subset of popular **NAS-Benchmark 101 dataset** and also evaluate the most popular
neural network architectures for two mobile GPUs. 



 To achieve this goal, we build open-source tools which provide a convenient way to conduct massive experiments on different target devices focusing on mobile GPU.

----------------这个具体是什么工具？？？？？



We implement lookup-table method which is used in **ProxylessNAS, ChamNet, FBNet** for prediction of latency on mobile CPU





#### A Study on Encodings for Neural Architecture Search

NASBench-101 (middle)

**Our results demonstrate that NAS encodings are an important design decision which can have a significant impact on overall performance.**   --- 不同的编码方式，来提升NAS的整体性能 。

**Sample-based NAS** is the most reliable approach which
aims at exploring the search space and evaluating the most promising architectures.
**However**, it is computationally very costly. As a remedy, the one-shot approach
has emerged as a popular technique for accelerating NAS using weight-sharing.
However, due to the weight-sharing of vastly different networks, the one-shot
approach is less reliable than the sample-based approach.

Our code is available at https://github.com/naszilla/nas-encodings

我们可以做： mobilenet V3的编码？？？



# NAS-Bench-101: Towards Reproducible Neural Architecture Search  

-- 提供accuracy 。。    -- google 

Recent advances in neural architecture search (NAS) demand tremendous computational resources, which makes it difficult to reproduce experiments and imposes a barrier-to-entry to researchers without access to large-scale computation. We aim to ameliorate these problems by introducing NAS-Bench-101, the first public architecture dataset for NAS research. To build NAS-Bench-101, we carefully constructed a compact, yet expressive, search space, exploiting graph isomorphisms to identify 423k unique convolutional architectures. We trained and evaluated all of these architectures multiple times on CIFAR-10 and compiled the results into a large dataset of over **5 million trained models.** This allows researchers to evaluate the quality of a diverse range of models in milliseconds by querying the pre-computed dataset. We demonstrate its utility by analyzing the dataset as a whole and by benchmarking a range of architecture optimization algorithms. 

---- 我们自己做accuracy 的数据集不现实， 如果要做latency 的数据集， 则要掌握部署一系列技能。



1. ![image-20201031115628817](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201031115628817.png)
2.  510M ,   5million （500W）model ,  423K  architectures.

#### NAS-BENCH-201: EXTENDING THE SCOPE OF REPRODUCIBLE NEURAL ARCHITECTURE SEARCH
Xuanyi Dong†‡ ∗and Yi Yang

In this work, we propose an extension to NAS-Bench-101: NAS-Bench-201 with a different search space, results on multiple datasets, and more diagnostic information. NAS-Bench-201 has a fixed search space and provides a unified benchmark for almost any up-to-date NAS algorithms.

dataset：  CIFAR-10, CIFAR-100, imagenet-16-120

Our NAS-Bench-201 is algorithm-agnostic.



![image-20201031133317635](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201031133317635.png)



  ([NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), [NAS-Bench-NLP](https://arxiv.org/abs/2006.07116)),  

#### NAS-BENCH-301 AND THE CASE FOR SURROGATE BENCHMARKS FOR NEURAL ARCHITECTURE SEARCH

1. We present NAS-Bench-301, a surrogate NAS benchmark that is first to cover a realistically-sized
search space (namely the **cell-based search space of DARTS** (Liu et al., 2019b)), containing
more than 1018possible architectures. This is made possible by estimating their performance via
a surrogate model, removing the constraint to exhaustively evaluate the entire search space.
2. We empirically demonstrate that a surrogate fitted on a subset of architectures can in fact model
the true performance of architectures better than a tabular benchmark (Section 2).
3. We analyze and release the **NAS-Bench-301** training dataset consisting of ∼60k fully trained and
evaluated architectures, which will also be publicly available in the Open Graph Benchmark (Hu
et al., 2020) (Section 3).
4. Using this dataset, **we thoroughly evaluate a variety of regression models as surrogate candidates**, showing that strong generalization performance is possible even in large spaces (Section 4).titive baseline on **our realistic search space.**
5. We utilize NAS-Bench-301 as a benchmark for running various NAS optimizers and show that
the resulting search trajectories closely resemble the ground truth trajectories. This enables sound
simulations of thousands of GPU hours in a few seconds on a single CPU machine (Section 5).
6. We demonstrate that NAS-Bench-301 can help in generating new scientific insights by studying
a previous hypothesis on the performance of local search in the DARTS search space (Section 6).





#### Chamnet -Towards efficient network design through platform-aware model adaptation   --- accuracy + latency + energy   code

----- by facebook and princeton 

We
formulate platform-aware NN architecture search in an op-
timization framework and propose a novel algorithm to
search for optimal architectures aided by efficient **accuracy**
and resource (**latency** and/or energy) predictors. At the core
of our algorithm lies an accuracy predictor built atop **Gaussian Process with Bayesian optimization** for iterative sampling. With a one-time building cost for the predictors, our
algorithm produces state-of-the-art model architectures on
different platforms under given constraints in just minutes.

![image-20201030094022752](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201030094022752.png)

对时延这个，我们是否可以做？？？？ 没有部署，是否可以真实反映时延 ？？？

 https://github.com/facebookresearch/mobile-vision 



#### FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search-- facebook  

![image-20201030110639229](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201030110639229.png)

search space ： layer wise 的， 类似mobilenet V2  ， shiftnet

![image-20201030112153408](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201030112153408.png)





#### BRP-NAS: Prediction-based NAS using GCNs

--- accuracy   +   latency 

GCN  + binary predictor 

NAS-Bench-101,  NAS-Bench-201 DARTS.  LatBench

transfer learning   (用延迟GCN的参数， 初始化精度GCN)

We also release Eagle which is a tool to measure and
predict performance of models on various systems. We make LatBench and the source code
of Eagle available publicly

不知道latency 的数据集是基于什么平台/ 后台度量工具来采的 。。。



**-------------------- 做proxyless 和MNAS的空间。 fastseg 搜索空间。 的编码， 用不同的模型来识别。**