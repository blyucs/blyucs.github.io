我的研究方向是：   人工 -- -NAS   意义 ，  自动化， 专门化， 定制化 ， 资源限制化。 

​    NAS 痛点  --- 算力  ， 效率提升。 



研究：

1. NAS 应用研究， 提升精度。 （图像分割应用的初探）
2. NAS 方向研究，效率提升 。 ---  各种方法。 （过渡到方法）
3. 平台化，定制化的NAS 搜索研究。  （方法研究）

面向工业化， 定制化的可落地的NAS 方法研究。 



![image-20201126111048887](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201126111048887.png)







#### 基于性能预测的神经网络架构搜索关键技术研究及应用

小：侧重于性能预测/推理时延预测的效果

**贡献：**：重点研究模型架构（包含各种搜索空间的，chain-style，nas-net，等全空间结构）的representation，研究预测模型， 赋能NAS， 提升NAS 过程的效率。---- 算是机器学习/深度学习方法的应用

1. 各种类型的结构，异构的网络结构的性能预测，时延预测 效果 ， **数据集的构建，公布**
2. 创新的编码方式 和 预测效果 
3. 创新的模型（Bi-LSTM, GNN/GCN, 集成学习方法等）应用 
4. 赋能NAS 后的效果（**跨平台，体系结构的效果**）



#### 高效神经网络架构搜索关键技术研究及应用

大：退可守的方向，比较广

**搜索过程的高效** ：  知识蒸馏/辅助分支/  性能预测 /  时延预测  / 超网络（once for all ）/ 架构可微  等技术， 从能源环境的角度谈论，

**搜索出的网络的高效**：resource-aware ， resource-constrained， 多目标的奖励函数，pareto-optimal

**贡献：**   

  1 . 性能预测 /  时延预测 方面提出一些方法 。  

2. 直接基于硬件的反馈，性能提升 效果 。 
3. 搜索策略上的创新  ？？？？

**不好点**：还是CNN？？ 内卷化的精度比拼。  

参考文献： 



#### 面向图神经网络的神经架构搜索技术研究及应用

专：进可攻的方向  +  可以和 “理论分析” “动力学建模” “复杂网络 ” 结合  。 

**贡献**（大）： ALL IS RL-based  ，  （可以用的是evolution ， random search）

1. 可以将NAS+CNN 上的经验往 NAS+GNN 过渡。



参考文献：

GraphNAS: Graph Neural Architecture Search with Reinforcement Learning    ---  **opensource**  中科院 IJCAI

Auto-GNN: Neural Architecture Search of Graph Neural Networks   

Simplifying Architecture Search for Graph Neural Network

Neural Architecture Search in Graph Neural Networks  ---  **opensource**   based on GraphNAS



A general deep learning framework for network reconstruction and dynamics learning   -  北师大，张江  





![image-20201126154139724](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201126154139724.png)