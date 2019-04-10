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
有两种方法赋值
1.tf.assign()
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
