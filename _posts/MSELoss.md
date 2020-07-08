

## MSELoss

MSELoss function which is widely used as a loss function of the regression problem creates a criterion  that measures the gap of  two vectors or two tensors. It is a element-wise operation，which can be descripted as follow：
$$
x={x_0,x_2......x_n}\\
y={y_0,y_1......y_n}\\
l(x,y)=\frac{1}{n}\sum^n_{i=1}(x_n-y_n)^2
$$

Let's leverage the MSELoss lib function of nn module in pytorch to illustrate this function:


```
input = torch.randn(3, 5, requires_grad=True) 
target = torch.randn(3, 5)
loss1 = nn.MSELoss(reduction='mean')  #the default para is 'mean'
loss2 = nn.MSELoss(reduction='sum')  #calc without the average operation
print(input)
print(target)
output1 = loss1(input, target)
output2 = loss2(input, target)
print(output1)
print(output2)
```

```
tensor([[ 0.8939, -0.1787,  0.0788, -0.7437, -1.6094],
        [-1.3230,  0.1581, -0.3811, -1.1255,  0.5299],
        [-0.5864,  1.3663, -1.2498,  0.9574, -0.6627]], requires_grad=True)
tensor([[-0.2026, -0.5270, -1.4095, -0.8290, -1.1343],
        [-0.9893, -0.1236,  0.8710, -0.6213, -1.0020],
        [-0.0906,  0.7872, -2.4900,  0.3343,  0.6884]])
tensor(0.8309, grad_fn=<MseLossBackward>)
tensor(12.4636, grad_fn=<MseLossBackward>)
```

Knowledge distillation

