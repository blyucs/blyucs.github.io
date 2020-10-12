![](D:\00_code\blyucs.github.io\images\MNASPP_exp\KD_MT.png)



2020.07.26:

1. 0.18 M 的ghostnet 预训练模型训练失败，文件夹被误删了，重新训练  

2. 32的batchsize， 训练10 epoch  ， loss 为  0.34 -- 暂时作为ecnoder - pre-trained  model  ---  结合当前的bachsize 设定：[32,64] , 训练速度和效果还可以 ，接近3min 一个epoch。

3. 发现搜索速度太慢， 打算放开 data -augment 中的操作。 **是否要在train的时候恢复回来 ？ ？？**应该是的

4. TODO : train 当中没一个epoch 保存一次模型。

   

5. 1.83M 的mobilenet + end to end 的预训练  ---已训练完成(  训练时要直接使用原始的图片构造dataloader，不需要进行chrop 和resize  这些数据增强操作。)， 可以达到 eyes/brows/overall 分别 88.8，86.0，95.5 的F1-score。  20200726T2312  -- 如果把这个模型直接改成not-end-to-end ，不改变权重， 则变为 83.3/81.7/95.1， 差距比较明显。

   但是使用改网络做知识蒸馏，出现kd_loss 过大的情况， 几千。 **估计重新不使用end-to-end 的训练  ?** 因为搜索过程不是end-to-end 的模型。search 过程的输出是64X64，  搜索过程输出是 512X512 , 在把参数复制到search 模型， 中间出了差错。 ---   **修改teacher 为 not-end-to-end 以后解决。**



2020.07.27:

1. kd net 和  pre-trained encoder 都准备好了， 基于KD 的正常搜索看起来还可以

2. 加上 polyak 和 aux 以后， 除了第一个架构正常训练外，其余后面的reward 全部为 0 --**待定位**

   ---- 确定试polyak 导致， 关于polyak 的代码， 同原文没有什么差别， 是否是改了decoder 的parallel 导致 ？??    ---  不是parallel 导致polyak 不行 --- 当前未确定原因。

3. 当前82 环境实行 基线搜索 （无任何惩罚） ， 19 环境实行 （param  0.4M/ -0.6 的搜索） 19： 20200728T1502    82： 20200728T1427



2020.07.28：

1. 除了第一个架构正常训练外，其余后面的reward 全部为 0 --已定位， 由于for 循环里面的采样架构后，没有加载预训练模型导致， 这将严重影响训练的收敛。
2. 尝试把branch =3， cells  = 2，  训练结构为eyes 78， brows 74.8  overall 90 ，效果不佳，这个是无aux 的训练，再加上aux 辅助， 效果也差不多。   **AUX 的意义 待验证。** 
3. 把 branch=4， cells=3 ， 相对于之前的减少了1对cells ，启动搜索，看看。----- 82 和 19 环境的搜索都呈现出，**两头多（0.6 ， 0.1 , 0.2 的很多），中间少的趋势（0.3， 0.4）**   ---尝试减少stage0 的训练次数为3次 ， 试试。 
4. branch=3， cells=3  ，0_num = 3, 1_num =1,   还是特别多0.6几的
5. branch=3， cells=2  ，0_num = 3, 1_num =1,   (为了压榨搜索，进一步压缩了空间 ？？ 而且这样将搜索过程数据打开来看的话， **也会更加呈现了架构的均匀分布性， 不会出现两头大，中间小这种情况** ， 或者优秀架构太多的情况。) 



2020.07.30:

1. 昨天进行的三个实验 ： 

   1)82搜索： 已经800+ epoch ，还是没有出现太多大于0.7 的架构 ，大部分集中在0.6几  。 **是否是架构过于简单导致 ？** （branch=3， cells=2  ，0_num = 3, 1_num =1, ）  --   是否尝试 （branch=4， cells=2  ，0_num = 3, 1_num =1,）

   2)82训练 ：eyes 86.0，  brows 83.4 ， overall 92.1   train end-to-end . 效果不佳，**如何兼顾搜索过程和最终效果 ？//**  --- 这个结果无效， 后面有重新训练

   3)19 parameter 搜索 ：有一定的压缩效果。但是**架构的表现的鉴别特性**很差。

   <img src="D:\00_code\blyucs.github.io\images\MNASPP_exp\19_para_comp_0729.png" style="zoom: 33%;" />

2. （branch=3， cells=3  ，0_num = 3, 1_num =1,） 恢复到这个配置再试试，依旧是0.6 以上的太多

4.  （branch=4， cells=4  ，0_num = 2, 1_num =1,）为了后面好提高精度，和 呈现压缩效果 -- 不太理想
5.  （branch=4， cells=3  ，0_num = 6, 1_num =1,）**恢复到这个？？？？**
6. 发现reward 的设置不合理， 不应该使用融合的f1-score, 这样不能真实反映效果，应该直接用单独的。-**-使用了单独的f1-score (l_eye,r_eye,l_brow,r_brow) , 这样得到了更好的效果，和分布划分。** 
7. 昨天的训练，进行了random mirror 等操作，不合理，这样丢失了左右属性。 昨天test 的效果整个eyes 很好，但是l_eye，r_eye 很差。 **新增了代码， 把stage 传入了dataloader** 。   ----  重新基于架构[8,[1,1,1,10],[4,3,1,3],[0,1],[2,0]] 进行20epoch 的训练 ， 得到了  88.4/84.5/94.5 的精度。
8. **将多任务训练代码onetrack ， 以便后续训练**
9. 当前启动了基于以上 的baseline 和  para 的训练 。 branch=3， cells=3  ，0_num = 6, 1_num =1



2020.07.31：

1. 对82 环境上搜索（branch=3， cells=3）的baseline 进行了分析， 发现 1） 参数量出现分层，就那么几种参数组合  2）reward 的增加时以参数量增加为代价获得的  ：**0730T1541**
<center>
<figure>
    <img src="D:\00_code\blyucs.github.io\images\MNASPP_exp\82_baseline_boxplot_0731.png" style="zoom:100%;" />
    <img src="D:\00_code\blyucs.github.io\images\MNASPP_exp\82_baseline_0731.png" style="zoom: 50%;" />
     </figure> 
</center> 

   原因： 1） 搜索空间太简单    2） 架构组合太简单（genotype）

2. 19 环境上的参数量搜索  （**0730T1811**  --  - 这个比baseline 快时因为架构相对更小）

<img src="D:\00_code\blyucs.github.io\images\MNASPP_exp\19_para_comp_0731.png" style="zoom:50%;" />

以上右下角区域，没有出现架构， 代表参数压缩失败。 

3.  由于架构的单一性和考虑后续训练精度，因此采用  branch=4， cells=3  ，0_num = 6, 1_num =1 ，基本恢复到了fastnas 的超参组合中。 

   进行了如下3个实验：

   1）82， （GPU 2,3）baseline 

   2）82， (GPU 0,1)   para   - 0.6

   3）19， （GPU 2,3,）PARA  -0.3



2020.08.01：

82_baseline_0801_scatter：
![82_baseline_0801_scatter](D:\00_code\blyucs.github.io\images\MNASPP_exp\0801\82_baseline_0801_scatter.jpg)
19_para_0.3_scatter_0801:
![19_para_0.3_scatter_0801](D:\00_code\blyucs.github.io\images\MNASPP_exp\0801\19_para_0.3_scatter_0801.jpg)
**outstanding 的点： 0.28M para ， original reward 0.66**
82_para_0.6_scatter_0801：
![82_para_0.6_scatter_0801](D:\00_code\blyucs.github.io\images\MNASPP_exp\0801\82_para_0.6_scatter_0801.jpg)
**outstanding 的点：**
82_baseline_boxplot_0801：
![82_baseline_boxplot_0801](D:\00_code\blyucs.github.io\images\MNASPP_exp\0801\82_baseline_boxplot_0801.jpg)
19_para_0.3_boxplot_0801:
![19_para_0.3_boxplot_0801](D:\00_code\blyucs.github.io\images\MNASPP_exp\0801\19_para_0.3_boxplot_0801.jpg)
82_para_0.6_boxplot_0801:
![82_para_0.6_boxplot_0801](D:\00_code\blyucs.github.io\images\MNASPP_exp\0801\82_para_0.6_boxplot_0801.jpg)

正交表，正交设计助手，判定表



2020.08.02：

1. 82-para-0.6 ： 跑完了 

2. 82-baseline 没有跑完

   

2020.08.03：
1. 82 — flops  -0.6： （base ==  1.4）-- 这个base 值太小了， 计划改为2.0

   ![](D:\00_code\blyucs.github.io\images\MNASPP_exp\0803\82_flops_0.6_0803.png)
太小的flops 是无法获得好的效果的。 如何避开这部分架构？？？？
为0 的架构，乘以系数以后，还是0 ， 但是如何避开那种本身只有0.18， 由于


2. 19 -- delay  -0.6 ：  这个base 值是0.8  ， 感觉还比较合适。   
3. 明天再针对flops 的以上情况。 新增一个分段的惩罚函数， 让断层的地方得到不同的处理。



2020.08.04：

1. 19-dela-0.6 的收敛比flops 要好。 

2. ![image-20200804220058549](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200804220058549.png)

3. flops 实验的baseline 对比，还没有。 需要把之前baline 采样出的对应架构，从中取出， 写个代码。 的flops 算出来。 拼接数据

   ![image-20200804192908152](D:\00_code\blyucs.github.io\images\MNASPP_exp\0804\image-20200804192908152.png)

大部分架构集中在无效的区域， 这将是非常浪费时间，且无效的。 同时，也要考虑到

3. 在82 环境上的delay 一直很大， 全部是0.7 以上的 ，而在19 环境上确比较小， 很多小于0.4/0.3 的 。 感觉以后都在19 上做delay 的实验/ 以及补充的baseline 实验 。 
4. 19 环境上进行了“避开过小flops”的代码（-0.6）， 直接对过小的不做奖励， 试试 。
5. 82 环境计划做delay的另一个参数（-0.3）的实验。  

![image-20200805220311297](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200805220311297.png)



2020.08.05：

1. 19 环境上进行了“避开过小flops”的代码（-0.6）， 直接对过小的不做奖励， 试试 。
2. 82 环境计划做delay的另一个参数（-0.3）的实验。  
3. 重启baseline 实验



2020.08.06：

1. 整理数据， 继续消融实验

2020.08.07 ：

1. 尝试遗传算法工具箱， 找pareto 点。 

   

![image-20200808105112232](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200808105112232.png)





2020.08.08：

params：



19_baseline_params:

![19_baseline_params](D:\00_code\matlab_mnaspp_scatter\0808\19_baseline_params.jpg)

19_params_03:
![19_params_03](D:\00_code\matlab_mnaspp_scatter\0808\19_params_03.jpg)

82_params_06:
![82_params_06](D:\00_code\matlab_mnaspp_scatter\0808\82_params_06.jpg)

pareto_params:
![pareto_params](D:\00_code\matlab_mnaspp_scatter\0808\pareto_params.jpg)

flops：
19_baseline_flops
![19_baseline_flops](D:\00_code\matlab_mnaspp_scatter\0808\19_baseline_flops.jpg)
82_no_drop_flops_06   --- **带来一个问题。集中到左下角**
![82_no_drop_flops_06](D:\00_code\matlab_mnaspp_scatter\0808\82_no_drop_flops_06.jpg)
19_with_drop_flops_06
![19_with_drop_flops_06](D:\00_code\matlab_mnaspp_scatter\0808\19_with_drop_flops_06.jpg)
pareto_flops_drop
![pareto_flops_drop](D:\00_code\matlab_mnaspp_scatter\0808\pareto_flops_drop.jpg)
82_with_drop_flops_03
![82_with_drop_flops_03](D:\00_code\matlab_mnaspp_scatter\0808\82_with_drop_flops_03.jpg)
pareto_flops
![pareto_flops](D:\00_code\matlab_mnaspp_scatter\0808\pareto_flops.jpg)

delay：
19_baseline_delay
![19_baseline_delay](D:\00_code\matlab_mnaspp_scatter\0808\19_baseline_delay.jpg)
19_delay_06
![19_delay_06](D:\00_code\matlab_mnaspp_scatter\0808\19_delay_06.jpg)
pareto_delay
![pareto_delay](D:\00_code\matlab_mnaspp_scatter\0808\pareto_delay.jpg)

2020.08.14：

1. 82 环境： 对972 ， 0.29 M 模型进行 only task1 摸底， 获得了 94.5. 
2. 82 环境：打算把task 0 和  KD 加上 试试，。  0814T1033---没有成功， GPU环境好像挂了 。

2. 19 环境：训练 9，1，0，10，3，4，0，3，9 .。 架构， loss 为0.27 ， 尚未测试。



2020.08.18 ：

1. 绘制采样图，
2. 把82 上面的代码同步到19 中， 进行测试用。 
3. 82 上面训练架构：  **2，0，1，2，7，4，2，9，9（para -0.6 972 ） ** 94.5
![image-20200820154612482](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200820154612482.png)
5. 19 上面训练：   **9，1，0，10，3，4，0，3，9 . （ para -0.3 456）**  -- 94.7

![image-20200820154702612](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200820154702612.png)



2020.08.19：



1.  从pareto 曲线中的点的颜色可以看出，更加优秀的架构是在一定的搜索迭代后才omit 出来的，这也进一步证明了搜索过程的有效性和必要性

2020.08.20

1. 19 ： flops - 805， 3，[0，0，10，3]，[3，3，10，9]     87.5/82.4/94.3

![image-20200822160738233](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200822160738233.png)

1. 82:   flops -653 ,  0,[1,0,3,7],[4,1,2,0]   88.4/84.8/94.7

![image-20200822160641588](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200822160641588.png)



2020.08.21：

1. 19 ： delay --980  --- **0.24** ： 9 [1，0，5，9]， [4，1，1，1]，  

   ![image-20200822160923976](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200822160923976.png)

2. 82. delay --**0.3**： 9，[1,0,1,3],[4,0,1,1]    

![image-20200822183548730](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20200822183548730.png)



2020.08.22:

1.  19 ： delay --980  --- **0.24** ： 9 [1，0，5，9]， [4，1，1，1]，    ----  考虑不使用  上采样的方式， 直接训练

0822T1634 得到 0.21 的loss 值， 测试结果： 83.8/81.0/ 93.9  证明end-to-end 确实差别很大



2. 进行batch training

2020.08.23：

1. batch training without kd   ， **在达到一定量的数据量下， 似乎， 有没有kd 都没有什么区别 。 好坏架构也不太区别得出来。 。。。。 ？???**   -----通过采样的架构的对比训练， 似乎， KD 并没有太大的好处。 但是two-stage 确实有更好的效果。

   **two-stage with kd 和  without kd 对比， 相同的epoch。** 

2. original_batch_training_validate   ： **也是用预训练模型**， 只是不采用pre-computing 和  two-stage 。 
    original  （with pre-trianed encoder ） 同 two stage without KD 的对比， 排除KD 的干扰。 

2020.08.25:

1..  type EG1800  ,   type-=  train, 有预训练模型， 不做two-stage训练， 只做train_segmenter. 可以正常训练 ， loss 值很低  0.08。 测试 92.9 ？？？

**但是测试的时候  512 ！= 800 不匹配。**  ---- 用resize 来解决。

2.   type EG1800  ,   type -=  search （not-end-to-end ）, 有预训练模型， 不做two-stage训练， 只做train_segmenter  --- 只有 69%的效果。





2020.08.26：

1. type = search  先进行stage 0  -  再stage 1 似乎效果很差， 可能和预训练模型有关， 预训练后， 再经过调整，反而调坏了。 
2. 直接只使用pre-trained + stage  0 ， type =  search ， resize 。  ---  直接过拟合（数据量太少了）。  loss 值很低， 但是表现不佳，80.7 . 
3. 直接 **no  pre-trained** + stage  1， type =  **train**， resize 。 相比于92.9 的， 没有使用pre-trianed model 。     ----  选了一个0.1 loss 值的架构来测试， 只有 80.4 ， 直观效果也很差， 是否是batch-size 为2 太小了。 