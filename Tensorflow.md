Thanks for Jacob Buckman's describe on https://jacobbuckman.com/post/tensorflow-the-confusing-parts-1/



Whats Tensorflow
======================

他是一个计算图，有向图

Graph
-------------

```bash
运行 import tensorflow as tf
```

得到的是一个空的图
```bash
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node ## equivalent to tf.add(two_node, three_node)
```
得到的是节点2和节点3指向节点+  并没有5

Session
-----------------

计算图相当于计算摸板，需要会话session去执行计算

```bash
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print sess.run(sum_node)
```
计算图上面依然是节点2和节点3指向节点+  并没有5，但是结果会输出5
！！！记住计算图只是模板
sess.run() 的调用往往是 TensorFlow 最大的瓶颈之一，因此调用它的次数越少越好。如果可以的话，在一个 sess.run() 的调用中返回多个项目，
而不是进行多个调用。

Placeholders & feed_dict
------------------------

占位符相当于用来接收输入

```bash
import tensorflow as tf
input_placeholder = tf.placeholder(tf.int32)
sess = tf.Session()
print sess.run(input_placeholder, feed_dict={input_placeholder: 2})
```
注意key和value的值

Computation Paths 计算路径
----------------

```bash
import tensorflow as tf
input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sum_node = input_placeholder + three_node
sess = tf.Session()
print sess.run(three_node)
print sess.run(sum_node)
```
print sess.run(sum_node)会报错，因为计算这个的时候发现input_placeholder没有值
```bash
 P   3
 |   |
 V   V
   +
```
计算路径是由下往上只包含必要节点，这样节省时间
比如print sess.run(three_node) 就只是3那个节点
print sess.run(sum_node)从+开始，往上走，发现3有值但P没有值，报错

Variables & Side Effects
-----------------
constant每次运行都一样
holder每次都不一样
variable可以改变

用变量表示参数，因为validattion的时候不需要改变
tf.get_variable(name，shape)
图上出现一个v节点，但是会报错因为没有值
有两种方法赋值 assign和initialize
1.tf.assign()
-----------

tf.assign(target, value) 

```bash
import tensorflow as tf
count_variable = tf.get_variable("count", [])
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
sess = tf.Session()
sess.run(assign_node)
print sess.run(count_variable)
```

![](https://github.com/ehamster/NLP/blob/master/images/assign.png)

tf.assign特殊性
1.不做运算
2.副作用计算路径经过的时候副作用发生在其他节点，这里是用zero_node值替换count_variable的值
3.非依赖边 count_variable and assign_node是独立的计算不会回流

计算路径流到任何点时也会执行该节点的所有副作用，这里是count_variable内存被永远设置0
下次再调用 sess.run(count_variable)不会报错而是得到0

2 initialize
-----------

```bash
import tensorflow as tf
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable("count", [], initializer=const_init_node)
sess = tf.Session()
print sess.run([count_variable])
```
v ...... const_init_node

这里只是将两个节点关联，变量count_variable内存仍然是null，并没有得到值
我们需要添加session使用const_init_node 更新变量的值

```bash
import tensorflow as tf
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable("count", [], initializer=const_init_node)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print sess.run(count_variable)
```
![](https://github.com/ehamster/NLP/blob/master/images/initialize.png)

添加了init = tf.global_variables_initializer()，当用session启动他时，所有initialize都会完成初始化赋值
在深度学习中，典型的「内循环」训练如下：

1. 获取输入和 true_output

2. 根据输入和参数计算「推测」值

3. 根据推测与 true_output 之间的差异计算「损失」

4. 根据损失的梯度更新参数

```python
import tensorflow as tf

### build the graph
## first set up the parameters
m = tf.get_variable("m", [], initializer=tf.constant_initializer(0.))
b = tf.get_variable("b", [], initializer=tf.constant_initializer(0.))
init = tf.global_variables_initializer()

## then set up the computations
input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x = input_placeholder
y = output_placeholder
y_guess = m * x + b

loss = tf.square(y - y_guess)

## finally, set up the optimizer and minimization node
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)

### start the session
sess = tf.Session()
sess.run(init)

### perform the training loop
import random

## set up problem
true_m = random.random()
true_b = random.random()

for update_i in range(100000):
  ## (1) get the input and output
  input_data = random.random()
  output_data = true_m * input_data + true_b

  ## (2), (3), and (4) all take place within a single call to sess.run()!
  _loss, _ = sess.run([loss, train_op], feed_dict={input_placeholder: input_data, output_placeholder: output_data})
  print update_i, _loss

### finally, print out the values we learned for our two variables
print "True parameters:     m=%.4f, b=%.4f" % (true_m, true_b)
print "Learned parameters:  m=%.4f, b=%.4f" % tuple(sess.run([m, b]))
```

optimizer = tf.train.GradientDescentOptimizer(1e-3)
这一行没有添加节点，只是创建python对象
train_op = optimizer.minimize(loss)
这一行添加了train_op节点伴随复杂的副作用：
train_op 回溯输入和损失的计算路径，寻找变量节点。对于它找到的每个变量节点，计算该变量对于损失的梯度。然后计算该变量的新值：
当前值减去梯度乘以学习率的积。最后，它执行赋值操作更新变量的值。

tf.Print
-------------

它有两个必需参数：要复制的节点和要打印的内容列表。他会输出input节点的copy同时，它的副作用是打印出「打印列表」里的所有当前值。

```python
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
print_sum_node = tf.Print(sum_node, [two_node, three_node])
sess = tf.Session()
print sess.run(print_sum_node)
```
[2][3]
5
注意：如果print这个节点不在计算流，他是不会打印的，即使他复制的点在计算流
最好在创建要复制的节点后，立即创建你的 tf.Print 节点。

举个栗子
-------

```bash
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
### this new copy of two_node is not on the computation path, so nothing prints!
print_two_node = tf.Print(two_node, [two_node, three_node, sum_node])
sess = tf.Session()
print sess.run(sum_node)
```

只会输出5，并没有 tf.Print(two_node, [two_node, three_node, sum_node]) 要打印的这些，因为
tf.print节点不在sess.run(sum_node)计算流上，即使他复制的two_node在。

![](https://github.com/ehamster/NLP/blob/master/images/tfprint.png)

Inspecting Graphs
==========

使用 tf.get_default()_graph() 返回一个指针指向 default global graph object

tf.Operation代表节点  tf.Tensor代表edge
create a new node:
1.我们收集与新节点的传入边相对应的所有tf.Tensor对象
2.我们创建一个新节点，它是一个tf.Operation对象
3.我们创建一个或多个新的传出边，它们是tf.Tensor对象，并返回指向它们的指针

有时候把node和tensor混为一体了，其实有点区别。具体可以看看 https://jacobbuckman.com/post/graph-inspection/

令人困惑的TF第二季！！
=================

https://jacobbuckman.com/post/tensorflow-the-confusing-parts-2/

Naming and Scoping
----------------

```bash
c = tf.constant(2., name="cool_const")
d = tf.constant(3., name="cool_const")
```
cool_const:0 cool_const_1:0
名字会默认不一样的

Using Scope
----------

with tf.variable_scope(scope_name):
可以给tensor分组

```bash
import tensorflow as tf
a = tf.constant(0.)
b = tf.constant(1.)
with tf.variable_scope("first_scope"):
  c = a + b
  d = tf.constant(2., name="cool_const")
  coef1 = tf.get_variable("coef", [], initializer=tf.constant_initializer(2.))
  with tf.variable_scope("second_scope"):
    e = coef1 * d
```
a.name = Const:0
c.name = first_scope/add:0 
e.name = first_scope/second_scope/mul:0

Sasving And Loading
---------------

work with single model:

 ```python
import tensorflow as tf
a = tf.get_variable('a', [])
b = tf.get_variable('b', [])
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')
 ```
最后产生四个文件
checkpoint
tftcp.model.data-00000-of-00001
tftcp.model.index
tftcp.model.meta

tftcp.model.data-00000-of-00001 包含模型权重（上述第一个要点）。它可能这里最大的文件。
tftcp.model.meta 是模型的网络结构（上述第二个要点）。它包含重建图形所需的所有信息。
tftcp.model.index 是连接前两点的索引结构。用于在数据文件中找到对应节点的参数。
checkpoint 实际上不需要重建模型，但如果在整个训练过程中保存了多个版本的模型，那它会跟踪所有内容。

saver.save(sess, './tftcp.model')
数值存在session中，所以要传入，

saver只会保存之前创建的所有variable 如果要选择保存
-----------------

saver = tf.train.Saver(var_list=[a,b])

Loading model
------------

```bash
import tensorflow as tf
a = tf.get_variable('a', [])
b = tf.get_variable('b', [])

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './tftcp.model')
sess.run([a,b])
```
不要初始化a b


注意：
1.模型保存了a b 读取的时候只有a没问题
2.模型保存了a 读取的时候有变量a b 会报错
3.模型保存了a 读取用的d 可以制定 a = d  saver = tf.train.Saver(var_list={'a': d})

```bash
import tensorflow as tf
a = tf.get_variable('a', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')

import tensorflow as tf
d = tf.get_variable('d', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list={'a': d})
sess = tf.Session()
sess.run(init)
saver.restore(sess, './tftcp.model')
sess.run(d)
```

模型检查
-------------

使用工具 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/framework/python/framework/checkpoint_utils.py 

```python
import tensorflow as tf
a = tf.get_variable('a', [])
b = tf.get_variable('b', [10,20])
c = tf.get_variable('c', [])
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.save(sess, './tftcp.model')
print tf.contrib.framework.list_variables('./tftcp.model')
```



常用函数
----------------------

```python
1. tf.argmax(input, axis)
axis = 0  得到每一列最大值下标
axis = 1  得到每一行的

2. tf.euqal(x,y)
如果x,y是数组，返回[True,True,False]这样

3.tf.cast(x,'float')
转化格式
[True,True,False]  变为[1.,1.,0.]

4.tf.reduce_mean()
求mean

5.aaa.eval() 等价于 sess.run(aaa)

aaa.eval()必须写在with Session() as sess: 里面
with tf.Session() as sess:
  print(accuracy.eval({x:mnist.test.images,y_: mnist.test.labels}))
  
6.tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
 input:[batch, in_height, in_width, in_channels] shape是这样的数据
 数量，高度，宽度，通道数
 filter：filter_height, filter_width, in_channels, out_channels 
 strides：卷积时在图像每一维的步长
 padding:只能是"SAME","VALID"其中之一 SAME是可以超出边界补0
 后面两个不用管
7.tf.nn.max_pool(value, ksize, strides, padding, name=None)
 value：和上面input一样
 ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
 strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

 
8.tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
 第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
 第二个参数labels：实际的标签，大小同上
 两个步骤： 1.输出的结果做softmax得到一个向量[Y1，Y2....Y10]
  2.和实际的Y做一个交叉熵运算[0,0,1,0,0,0,0,0]  
  结果越小越准确，因为H(y) = - sum(yi`log(yi))  yi`就是实际标签上第i个值，yi是预测第i个值
  
  如果是求loss，就不求sum而是求mean

9.
#其效果和下面的代码是等价的

with tf.Session() as sess:
  print(sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels}))
#eval()只能用于tf.Tensor类对象，也就是有输出的Operation。对于没有输出的Operation, 可以用.run()或者Session.run()。Session.run()没有这个限制


```
