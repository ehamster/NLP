1.学习指数衰减法
====================

exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

通常在训练的时候建议学习率衰

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\LARGE&space;learning\_rate*decay\_rate^{\frac{global\_step}{decay\_steps}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;\LARGE&space;learning\_rate*decay\_rate^{\frac{global\_step}{decay\_steps}}" title="\LARGE learning\_rate*decay\_rate^{\frac{global\_step}{decay\_steps}}" /></a>

```bash
参数：

learning_rate - 初始学习率
global_step - 用于衰减计算的全局步骤。 一定不为负数。喂入一次 BACTH_SIZE 计为一次 global_step
decay_steps - 衰减速度，一定不能为负数，每间隔decay_steps次更新一次learning_rate值
decay_rate - 衰减系数，衰减速率，其具体意义参看函数计算方程(对应α^t中的α)。
staircase - 若 ‘ True ’ ，则学习率衰减呈 ‘ 离散间隔 ’ （discrete intervals），具体地讲，`global_step / decay_steps`是整数除法，
衰减学习率（ the decayed learning rate ）遵循阶梯函数；若为 ’ False ‘ ，则更新学习率的值是一个连续的过程，每步都会更新学习率。
返回值：

与初始学习率 ‘ learning_rate ’ 相同的标量 ’ Tensor ‘ 。
 优点：

训练伊始可以使用较大学习率，以快速得到比较优的解。
后期通过逐步衰减后的学习率进行迭代训练，以使模型在训练后期更加稳定。 
```




