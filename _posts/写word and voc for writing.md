Recently , I've been connecting with just these partner id, e.g. 1350148235 ,1399687179 , and 1391982535, which are all server of our LAB. No more than 10 partner id have i connected in history.  I'm a PHD student in UESTC . I'm using teamviewer for my personal research work, not commercial use.  I do appreciate the the convenience that come form teamviewer and have been recommending it to my friends.  Please analyze my usage history  and reset my account free, thanks. 

Semantic segmentation can be formulated as a dense prediction task. The goal is to predict for every pixel which object class it belongs to.

Semantic image segmentation with high accuracy and eﬃciency using convolutional neural networks has been a popular research topic in computer vision


 In contrast to the aforementioned areas,

More crucially, we demonstrate how to efﬁciently search for these architectures within limited time and computational budgets. 

More crucially, we demonstrate how to efﬁciently search for these architectures within limited time and computational budgets. 

Potentially,this may well mean that instead of manually adapting a single stateof-the-art architecture for a new task at hand, the algorithm would discover a set of best-suited and high-performing architectures on given data
使用手头现成的SOTA的架构，不一定能够在特定的数据集上取的较好的效果，因此，为避免花费大的精力用于架构设计，NAS 出场。可以发现一系列具有较好效果的网络架构。

To this end,
We restrict our attention to the decoder part,as it is currently infeasible to perform a full segmentation network search from scratch

their summation serves as the cell output

At the very least： 至少

At this point, it is important to emphasise the fact that such accomplishments required an excessive amount of computational resources


Thus, still aligning with the goal of having a compact but accurate model, we explicitly aim to ﬁnd ways of performing steps that are beneﬁcial during training and obsolete during evaluation


Nonetheless   尽管如此

a variety of high level applications, such as face understanding, editing and animation

general scenes.  通用场景

that is 也就是说 

Throughout this work

Our goal is to 我们的目标是

Our algorithm compares favorably with all previous works on all datasets evaluated

exploited，  resort to 求助于，利用  As shown later in the ablation study, we resort to a better heuristic rule.

a series of / a range of  一系列

conducted a huge number of experiments  ： 做了大量的实验

use/employ /utilize/resort/leverage/perform(执行)   conducted a huge number of experiments

intend to/would like to

aforementioned: 前面提到的  As aforementioned

intentionally: 故意的

More crucially： 更关键的

iteratively ： 迭代的，反复的

This is no longer the case as  ----- is fast-growing.:随着事务的发展，xx 的情况已不再如此

This is no longer the case as the automated neural architecture search - a way of predicting the neural network structure via a non-human expert (an algorithm) - is fast-growing.

Potentially, this may well mean that  这可能意味着。。。。NAS 比人工更好等等

explicitly  ： 明确的

Concretely,  具体地
empirical： 经验性地  In this work, we empirically find that uniform sampling is good enough
Along with empirical results, this is the primary motivation behind the described approach.

Now, we have reached the stage

While there is a lack of theoretical work behind this latter approach, several promising empirical breakthroughs have already been achieved

At this point, it is important to emphasise the fact that

at the expense of ： 以... 为代价

To this end

In terms of NAS in semantic segmentation ： 在NAS 的语义分割方面

We primarily focus on two research questions 

In a similar vein of research ： 类似的研究

whereas “meta-val”, on the other hand,：然而，另一方面

heuristic ： 启发  We exploit a simple heuristic ， Concretely

introduce /propose/ identified（鉴定，指出观点） 

get / achieve / receive 获得

We further look for ways of：我们进一步寻求。。。方法

omitted： 忽略的  emitted ： 选出，射出

conduct/ do/  ： 做 实验等

outperform/surpass/exceed

method/ techniques /algorithm /approach  ： 方法，算法

straightforward: 坦白的，直白的

As evident from it ： 很明显， 展示图例、表格的时候用

For simplicity ：简单起见

poor-performing and well-performing architectures

showcased ： 展示了 

respectively.  ： 分别的

xx 在xx 领域展现了巨大的潜力： 如何表达 ？ ？ ？

adopt ： 把xx 引用到xx，移植 

we empirically found that 

A directed acyclic graph (DAG) is used to represent the network topology architecture

The most similar work ， 

drawback ： 缺点

a large proportion : 很大比例

endeavored to  ： 努力做到

inherently： 固有的

Notably, ： 显著的

Critically, 

Indeed, 事实上

literature ： 文献

premise： 前提

multi-scale： 多尺度

3 × 3 atrous separable convolution： 分离式空洞卷积

网络描述：
For the spatial pyramid pooling operation, we perform average pooling in each grid. After the average pooling, we apply another 1 × 1 convolution followed by bilinear upsampling to resize back to the same spatial resolution as input tensor. For example, when the pooling grid size gh × gw is equal to 1 × 1, we perform image-level average pooling followed by another 1 × 1 convolution, and then resize back (i.e., tile) the features to have the same spatial resolution as input tensor.
We employ separable convolution [79, 85, 86, 17, 38] with 256 filters for all the convolutions, and decouple sampling rates in the 3 × 3 atrous separable convolution to be rh × rw which allows us to capture object scales with different aspect ratios. See Fig. 2 for an example.

To deal with these issues, /  

on the order of ：... 左右

inordinate ： 过度的

low resolution images ： 低分辨率图像

but is predictive of larger tasks  ： 但是对大型任务有预测性

diagram ： 简图， 如表示搜索架构的图

human-invented architectures ： 人工设计网络

deploy : 部署

atrous spatial pyramid pooling (ASPP) ： 

merit（优点）/advantage   

to find a tradeoff between speed and performance for: 找到一个权衡

CAS jointly learns the architecture of the cells as well as the associated weights ； jointly :联合的

thoroughly： 彻底地

Namely, 也就是说

Momentum and weight decay are set to 0.9 and 0.0005, respectively

general ： 一般

atrous spatial pyramid pooling: 空洞空间金字塔池化   -- form deeplab V3

spatial pyramid pooling ： 空间金色塔池化

Hierarchical ： 分层的。 

would not suffice for   不够的

controls/ govern

problematic for： 对XX 是有问题的

in the interests of time： 为了节省时间

interpolate  ： 插值 


a computing cell with/without constraints. Each edge represents one operation between two nodes. The top graph shows many candidate operations existing between nodes, and each candidate operation has its own cost.


%参照CAS 论文的方法，画一个摘要示意图，把搜索框架和任务都加进去

numerous/ a variety of / widespread： 大量的

there exist many types of data errors (e.g. redundant（多余的）, incomplete（不完全的）, or incorrect（不正确的） data)

作名词用的 state of the art，可以翻译成「（最）前沿水平」或「最高水平」。

作形容词用的 state-of-the-art，可以翻译成「（最）前沿的」。

time-consuming  ： 耗费时间的

Likewise, 同样的（ 句子开始）  likely ： 很可能

train RetinaNet to converge : 训练到收敛

focus on / concentrate on / devote to /commit to/ dedicate to  致力于

the former ... the later ...

He pioneered the idea of the corporation as a social institution.

inferior(低等的)  superior （高等的）


creatively/first /pioneer/ groundbreaking 

moreover/ futhermore / besides

pertaining to  适合

up to two orders ofmagnitude  最多两个数量级


modalities  形态  

and vice versa  反之亦然  

A vast body of literature  ：  大量的文献

in the interests of ： 为了， 考虑到

tailoring ： 调整

Digging deeper  ： 深入调查 

is termed as ： 被叫做

The potential reason is that, finding architectures with extremely low latency  潜在的原因是。。。 	

In terms of ： 在,...方面

Example of the macro architecture being sampled process. A RL Controller (bottom) sequently generates connections between encoder feature maps and decoder cells, or between the frontier node output layers and the latter cells. For this instance, the controller first samples indices cell4_0 and cell4_1. Both indices stand for the encoder feature layers, which would be fed into the corresponding cells, before being summed up to create node4. All the node output layers are fed into convolution followed by the final classifier.

resource-aware NAS  ：  加入
constrainted macro architecture   

substantial  大量的

 are the de facto ：  实际上是

have turned the problem inside out   彻底解决了这个问题

labelled  贴标签的

alleviate ： 减轻

account for : 对xxxxx 负责

dominate： 比什么 更好

balance exploration and exploitation

depict ： 描述

having the number   .... would   : 如果， 那么 having would 

we consider the geometric mean of three quantities: namely  ：也就是说

mimic ： 模仿
in the sense that ： 在某种意义上， 就。。 而言， if 
depict  ： 描述

with respect to : 关于

aligning with 和。。。 一起 ???

given ： 考虑到

so as to : 从而

w.r.t   with respect to 关于，谈及，涉及到，   i.e.  id est ： 也就是    e.g. 例如
we use the gradient w.r.t. binary
gates to update the corresponding architecture
parameters

motivated by  ,  inspired by  ： 受启发 。   inspire us to      motivate us to do

given:考虑到  

In light of this ： 鉴于此， 考虑到 

ever-increasing ： 持续增长的

so as to , so that , thereby  : 因此， 从而。 

conditioned on / according to  : 根据

drastically ： 彻底的，激烈的。 

hand-tune : 手动调整。 

hardware-centric ： 以硬件为中心。

used to be: 过去曾经是

negligible: 微不足道的。

tens of thousands of： 成千上万的

outperform  超过

many orders of  多个数量级

nested  嵌套的

ongoing ： 正在进行的

For clarity  为清晰起见

tranductive  ： 直推式   and    inductive：归纳式

facilitate  促进

Ideally, 理想情况下

In that regard :  在这方面

Put simply,    简单的说

Aiming to alleviate this common issue,  为了缓解这个问题

While there was a big emphasis on ： 虽然非常重视 。 

In essence： 在本质上

shift from... to .... 从... 转移到。。。

intervention： 介入

promising 有前途的

 This has opened up a path towards computationally feasible architecture search  这为算力可行的架构搜索指明了道路。

 discrepancy ： 矛盾，差距

affect  influence  ： 影响

STGSN with single-head and multi-head attention has not much difference **w.r.t** performance

 is non-trivial due to three major challenges as follows: 并非易事，由于。。。

abovementioned ： 上面提到的

Overall, I think both frameworks have their merits.