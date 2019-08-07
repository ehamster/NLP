1.Transfrom模型
==================

```bash
1.由n个编码器，n个解码器和连接部分组成
  a)编码器不共享参数，结构一样。
    由两部分组成：self-attention 和feedforward网络
  b)解码器
    由三部分组成：self-attention，编码解码注意力层 和feedforward网络
    
    
2.过程
  a)转为词向量
  b)将一句话的词向量输入编码器的self-attention和feedforward层，自注意层有前后依赖关系，ff层没有，所以可以并行输入
  ff网络是相同的，每个词的词向量可以并行通过


3.计算自注意力
  a)每个编码器的输入向量（每个单词的词向量）中生成三个向量：查询向量、一个键向量和一个值向量，通过词嵌入与三个权重矩阵后相乘创建的
  b)计算得分
  
```

self-attention层计算过程
![](https://github.com/ehamster/NLP/blob/master/images/attention.png)



```bash
4.实际运行是以矩阵的形式跑
5.多头注意力 multi-headed attention 机制，生成多个 q k v
按照上面的过程，也会生成多个z  把多个z连接起来 * w0  得到一个最终的Z


```

![](https://github.com/ehamster/NLP/blob/master/images/multi.png)


```bash
6.为了记录位置信息，word embedding之后先加一个位移矩阵，且每一个子层周围都有一个归一化
```

![](https://github.com/ehamster/NLP/blob/master/images/%E7%BC%96%E7%A0%81%E5%99%A8.png)


1.2解码器
------------------

