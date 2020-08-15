#### 方式一

Init:

self.op1 =  ctx_cell()

self.op2 = ctx_cell()

self.agg = Aggregatecell()

Forward: 

x1= self.op1(x)

x2 = self.op2(x)

return self.agg(x1,x2)

#### nn.Sequential

A sequential container. Modules will be added to it in the order they are passed in the constructor. Alternatively, an ordered dict of modules can also be passed in.

一个有序的容器，**神经网络模块**将按照在传入构造器的顺序依次被添加到计算图中执行，同时以**神经网络模块为元素的有序字典**也可以作为传入参数。
# Example of using Sequential
```python
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )

    # Example of using Sequential with OrderedDict
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))
```
#### OrderdDict

有序字典： 键值对

Pytorch 里每个module 对象里的子module（_modules）都会以OderdDict 的形式存在。

如果，子module有命名则以“命名”：“内容存在” 。 可能是 **成员名**（self.name）,或是 的名字（add_module("xxxx", conv)） ,或是 squential（OderdDict(['CONV1', nn.Conv2d])） 初始化的名字。

否则，以“1”：“内容”， “2”："内容" ，默认1，2，3 的形式存在。

![](D:\00_code\blyucs.github.io\images\pytorh_module\orderdict.png)



#### nn. ModuleList 

传入一个[] , 数组， 

```
cells = []
aux_clfx = []
cells.append(module)
aux_clfx.append(module)
self.cells = nn.ModuleList(cells)
self.aux_clfs = nn.ModuleList(aux_clfs)
```

#### setattr/getattr

为对象属性赋值

```
setattr(
self, 
'adapt{}'.format(out_idx + 1), 
conv_bn_relu(size, agg_size, 1, 1, 0, affine=True)
)
```

获取对象属性的值

```
x[out_idx] = getattr(
self, 
'adapt{}'.format(out_idx + 1)
)(x[out_idx])
```

示例代码



```python
class MicroDecoder(nn.Module):
    """
    Parent class for MicroDecoders
        l1, l2, l3, l4, None - pool of decision nodes Decoder config must include:
     cell config
     a list of aggregate positions (can be identical)

    in the end, all loose connections from modified layers
    must be aggregated via the concatenation operation
"""

def __init__(
        self,
        inp_sizes,
        num_classes,
        config,
        agg_size=64,
        num_pools=4,
        ctx_cell=ContextualCell,
        aux_cell=False,
        repeats=1):
    super(MicroDecoder, self).__init__()
    cells = []
    aux_clfs = []
    self.aux_cell = aux_cell
    self.collect_inds = []
    ## for description of the structure
    self.pool = ['l{}'.format(i + 1) for i in range(num_pools)]
    self.info = []
    self.agg_size = agg_size

    ## NOTE: bring all outputs to the same size
    for out_idx, size in enumerate(inp_sizes):
        setattr(self,
                'adapt{}'.format(out_idx + 1),
                conv_bn_relu(size, agg_size, 1, 1, 0, affine=True))
        inp_sizes[out_idx] = agg_size

    inp_sizes = inp_sizes.copy()
    cell_config, conns = config
    self.conns = conns
    self.ctx = cell_config
    self.repeats = repeats
    self.collect_inds = []
    self.ctx_cell = ctx_cell
    for block_idx, conn in enumerate(conns):
        for ind in conn:
            if ind in self.collect_inds:
                # remove from outputs if used by pool cell
                self.collect_inds.remove(ind)
        ind_1, ind_2 = conn
        cells.append(MergeCell(cell_config, conn,
                               (inp_sizes[ind_1], inp_sizes[ind_2]),
                               agg_size,
                               ctx_cell, repeats=repeats))
        aux_clfs.append(nn.Sequential())
        if self.aux_cell:
            aux_clfs[block_idx].add_module(
                'aux_cell', ctx_cell(self.ctx, agg_size, repeats=repeats))
        aux_clfs[block_idx].add_module(
            'aux_clf', conv3x3(agg_size, num_classes, stride=1, bias=True))
        self.collect_inds.append(block_idx + num_pools)
        inp_sizes.append(agg_size)
        ## for description
        self.pool.append('({} + {})'.format(self.pool[ind_1], self.pool[ind_2]))
    self.cells = nn.ModuleList(cells)
    self.aux_clfs = nn.ModuleList(aux_clfs)
    self.pre_clf = conv_bn_relu(agg_size * len(self.collect_inds),
                                agg_size, 1, 1, 0)
    self.conv_clf = conv3x3(agg_size, num_classes, stride=1, bias=True)
    self.info = ' + '.join(self.pool[i] for i in self.collect_inds)
    self.num_classes = num_classes
    # self.upsample_x = nn.ConvTranspose2d(48, 48, 3, 1, 0, bias=False)

def prettify(self, n_params):
    """ Encoder config: None
        Dec Config:
          ctx: (index, op) x 4
          conn: [index_1, index_2] x 3
    """
    header = '#PARAMS\n\n {:3.2f}M'.format(n_params / 1e6)
    ctx_desc = '#Contextual:\n' + self.cells[0].prettify()
    conn_desc = '#Connections:\n' + self.info
    return header + '\n\n' + ctx_desc + '\n\n' + conn_desc

def forward(self, x):
    x = list(x)
    aux_outs = []
    for out_idx in range(len(x)):
        x[out_idx] = getattr(self, 'adapt{}'.format(out_idx + 1))(x[out_idx])
    for cell, aux_clf, conn in zip(self.cells, self.aux_clfs, self.conns):
        cell_out = cell(x[conn[0]], x[conn[1]])
        x.append(cell_out)
        aux_outs.append(aux_clf(cell_out.clone()))

    out = x[self.collect_inds[0]]
    for i in range(1, len(self.collect_inds)):
        collect = x[self.collect_inds[i]]
        if out.size()[2] > collect.size()[2]:
                # upsample collect
        collect = nn.Upsample(
             size=out.size()[2:], mode='bilinear', align_corners=False)(collect)
        elif collect.size()[2] > out.size()[2]:
            out = nn.Upsample(
                size=collect.size()[2:], mode='bilinear', align_corners=False)(out)
            out = torch.cat([out, collect], 1)
   
    out = F.relu(out)
    out = self.pre_clf(out)
    out = self.conv_clf(out)
    return out, aux_outs
```


