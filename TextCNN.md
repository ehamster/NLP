CNN for NLP
===================

1.what is convolution
--------------------
```bash
对于一个矩阵进行滑动的window function
在nlp中，因为每一个对象(比如一个句子)，都是word embedding转换为一个矩阵x*y
所以用一个window 扫描这个矩阵的时候，这个window的宽必须和y一样，x可以不一样，这样从上往下扫描
不同的widows只是x不一样。
下图
```
![](https://github.com/ehamster/NLP/blob/master/images/TextCNN.png)


2.Narrow or wide convolution
--------------
```bash

不够的地方添加0为 wide 不添加为narrow
```

3.pooling
---------------
```bash
一般是max poolling ,目的是获取固定size的输出，确实这样会丢失global info
就是这几个词出现的位置，但是不会丢失ngram信息， 比如还是获取了 not amazing 和 amazing not 的区别
```

4.channel 
---------
```bash
类似于图片有RGB，nlp类似于有不同weight
```

5.conclution
-----------
```bash
优秀：擅长文本分类，情例如感分析，垃圾邮件提取，主题分类
可以不需要做word2vec 模型直接会做onehot
缺点:不擅长POS tagging和试题提取，因为pooling操作丢失了word global info
可调整的部分：
1.input representations(word2vec,onehot等)
2.filter的数量和长度
3.pooling是max或者average
4.激活函数relu或者tanh等
等
```

[1] Blog:http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
[2] Github: https://github.com/dennybritz/cnn-text-classification-tf
