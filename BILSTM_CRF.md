1.BILSTM
====================

2.'CRF' LAYER
====================

1) Why do not choose softmax?
---------------------------

![](https://github.com/ehamster/NLP/blob/master/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20190430145834.png)

softmax 只是将数据归一化为(0,1)范围内且和为1,并没有约束顺序的作用，所以会预测出非法顺序

2) Use CFR Layer
-----------------------

CRF can learn constrain from training data
CRF 设置了转移矩阵，意味着两个标签位置顺序的概率(添加start end标记)

![](https://github.com/ehamster/NLP/blob/master/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20190430150829.png)

假设一个句子有N种序列可能性：

![](https://github.com/ehamster/NLP/blob/master/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20190430151243.png)

P(Total) = P1 + P2 + PN = e^{S1} + e^{S2} + e^{S3}

具体可以参考
https://createmomo.github.io/2017/10/17/CRF-Layer-on-the-Top-of-BiLSTM-4/

