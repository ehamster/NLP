模型学习方法，会其意，知其形。先动他的目的的理念，再看公式
================
![](https://github.com/ehamster/NLP/blob/master/images/conclusion.jpg)

最大似然估计
--------------

就是使用一种方法估算出一件事情发生的概率，也就是那个参数
比如：我们要求命中靶心的概率p = 0.5，但是我们只有实验结果并不知道这个值
所以我们做三次实验，射10次，分别命中的次数是4，5，6
要使连乘的值最大： p(4) * p(5) * p(6)   意味着要找到一个参数，使同时发生这三种情况的概率最大
因为 事实就是因为p=0.5才 发生了这三件事，我们只要找到让连乘最大的参数，就是p的近似值

监督学习的模型分为 判别式模型(discriminative model) 和生成式模型(generative model)
------------
```python
自己的理解:
生成式模型是连和概率建模，而且假设条件都是独立的，比如naive bayes 而且状态序列y决定观测序列x(颇有先验概率的感觉)
判别式模型比如逻辑回归，对条件概率建模，认为观测状态x决定状态序列y，颇有后验的感觉
假设你现在有一个分类问题，x是特征，y是类标记。用生成模型学习一个联合概率分布P（x，y），而用判别模型学习一个条件概率分布P（y|x）。
用一个简单的例子来说明这个这个问题。假设x就是两个（1或2），y有两类（0或1），有如下如下样本（1，0）、（1，0）、
（1，1）、（2，1）
则学习到的联合概率分布（生成模型）如下：-------0------1
                               ------1-- 1/2---- 1/4     
                               ------2-- 0 ------1/4

而学习到的条件概率分布（判别模型）如下：-------0------1
                               ------1-- 2/3---  1/3 
                              -------2--  0---     1




判别式模型：
1.对 P(Y|X) 建模
2.对所有的样本只构建一个模型，确认总体判别边界
3.观测到输入什么特征，就预测最可能的label
4.另外，判别式的优点是：对数据量要求没生成式的严格，速度也会快，小数据量下准确率也会好些。

同样，B批模型对应了生成式模型。并且需要注意的是，在模型训练中，我学习到的是X与Y的联合模型p(Y|X)  ，也就是说，
我在训练阶段是只对P(Y|X)建模，我需要确定维护这个联合概率分布的所有的信息参数。完了之后在inference再对新的sample计算  ，
导出  ,但这已经不属于建模阶段了。
生成模型：
1.对P(X,Y)建模
2.这里我们主要讲分类问题，所以是要对每个label（ yi）都需要建模，最终选择最优概率的label为结果，所以没有什么判别边界。
（对于序列标注问题，那只需要构件一个model）
3. 中间生成联合分布，并可生成采样数据。
4.生成式模型的优点在于，所包含的信息非常齐全，我称之为“上帝信息”，所以不仅可以用来输入label，还可以干其他的事情。
生成式模型关注结果是如何产生的。但是生成式模型需要非常充足的数据量以保证采样到了数据本来的面目，所以速度相比之下，慢。


```
1.最大熵模型：
---------------
  分担风险，考虑所有可能，但是实现复杂效率低
  
  
2.HMM隐马尔科夫： 生成式模型，来自朴素贝叶斯
--------------

```bash
 简单的说,HMM就包含3个参数  Lambda = {A,B,Pi}
 图模型：
 
 Hidden：  Qt-1 ->  Qt  -> Qt+1
             |      |       |
             v      v       v
 Observe:    Yt-1   Y       Yt+1
 
 观察序列是相互独立的 
 A： 是转移概率的矩阵    transition Prob P(qt|qt-1)   Q1 -> Qn  n = 1 ~ N  Pn的和为1  
 B:  是发射概率的矩阵    emission Prob   P(yt|qt)    发射概率和也为1
 Pi: 初始值  P(Q1)
  
 
 
 ```
   
   🌰：
   --------
   ```bash
   x:         john saw the saw
              |     |   |   |
   y: start-> PN    V   D   N ->end
   
   p(x,y) = p(y)p(x|y)
   
   P(y) = P(PN|start)*P(V|PN) * P(D|V) * P(end|N)
   P(x|y) = P(john|PN) * P(saw|V) * P(the|D) * P(saw|N)
   HMM只考虑1gram，这就是他的弊端
   
   假设有十个人每个人都说cat,输入, 使用最大似然估计得到Lambda(cat)  就是结果为cat的参数
   都说dog，得到lambda(dog) 
   这时候第11个人说一个词，如何判断是cat还是dog？
   
 得到P(Y11|lambda(cat))   和  P(Y11|lambda(dog))
哪个大就是哪个。  因为他会每一种可能都得到参数,所以叫生成模型
```
   ![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-03-15%20at%2014.57.36.png)
   
   
   
   
   3.CRF 条件随即场  conditional random field
   ------------------
   ```bash
   最简单版本：线性的。
   特征函数接受4个参数:  返回0或者1
   1.句子s
   2.第i个词
   3.第i个词的词性
   4.第i-1个词的词性
   
   许多特征函数得出评分，就能得出句子的probability
   λ是一个系数
   ```
![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-03-18%20at%2009.23.44.png)

4.Word embedding
-----------------

```bash
相关概念
--------------
word Embedding:
把单词或短语映射成实数向量
One Hot encoding:
最简单的word embedding，对于词A在他的位置置1，其他位置置0
N-gram:
基于假设第n个词的出现与前n-1个词相关。
  作用：
  1.计算ngram距离：
  两个字符串s(begin A B C end)和t(begin A B end)
  采用N=2的二元模型  s:(begin A) (A B) （B C） (C end)
  t: (begin A) (A B) (B end)
  distance = G(s) + G(t) - 2 G(s)&G(t) = 4 + 3 - 2（2） = 3
  2.判断句子可能性
  对于s p(s) = p(A|begin)p(B|A)p(C|B)p(end|C)
  3.用在特征工程，可以将P(A|begin）等作为新特征
先将词变为onehot，比如一个 1* 10000的向量，然后和模型的参数矩阵10000*300相乘,就会得到 1*300的vector
还有其他word embedding方法，比如共现矩阵，distributed representation等

```
Word2Vec
---------------
他是word embedding的一种
一共两种：
1.如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『Skip-gram 模型』
2. 而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『CBOW 模型』

```bash
1.CBOW  continuous Bag of words
eg  CBOW根据上下文求中心词
上下文的onehotencoding * w 就得到word2vec。  worc2vec * w‘ 得到中心词
比如，在skipgram来说：
  input: 1xV one hot encoding   假设一共V个词
  hidern layer : VxN W  N远远小于V
  output: 1XN * NxV W'  = 1XV  softmax
  输出层计算量太大，需要计算所有词的softmax得分，所以word2vec有2种优化
```
![](https://github.com/ehamster/NLP/blob/master/images/cbow.png)

假设上下文长度4，如图
这样我们这个CBOW的例子里，我们的输入是8个词向量，输出是所有词的softmax概率（训练的目标是期望训练样本特定词对应的softmax概率最大），对应的CBOW神经网络模型输入层有8个神经元，输出层有词汇表大小个神经元。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。

2。Skip_Gram
-----------
```bash
Skip-Gram模型和CBOW的思路是反着来的，即输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量。还是上面的例子，我们的上下文大小取值为4，
特定的这个词"Learning"是我们的输入，而这8个上下文词是我们的输出。

　　这样我们这个Skip-Gram的例子里，我们的输入是特定词， 输出是softmax概率排前8的8个词，对应的Skip-Gram神经网络模型输入层有1个神经元，
输出层有词汇表大小个神经元。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。
这样当我们有新的需求，要求出某1个词对应的最可能的8个上下文词时，
我们可以通过一次DNN前向传播算法得到概率大小排前8的softmax概率对应的神经元所对应的词即可。
总结：
隐层没有使用任何激活函数，但是输出层使用了sotfmax。
我们基于成对的单词来对神经网络进行训练，训练样本是 ( input word, output word ) 这样的单词对，input word和output word都是one-hot编码的向量。
最终模型的输出是一个概率分布。  

我们提到，input word和output word都会被我们进行one-hot编码。仔细想一下，我们的输入被one-hot编码以后大多数维度上都是0（实际上仅有一个位置为1），
所以这个向量相当稀疏，那么会造成什么结果呢。如果我们将一个1 x 10000的向量和10000 x 300的矩阵相乘，它会消耗相当大的计算资源，为了高效计算，
它仅仅会选择矩阵中对应的向量中维度值为1的索引行（这句话很绕）
但是实际word2vec不是这样使用DNN,而是进行优化，因为这样太慢了
隐藏层: [1* 10000 ] * [10000*300] = [1*300] 实际上没做矩阵乘法而是直接找1的位置然后找权重矩阵对应的行
输出层就是softmax，每个节点得到输出的值（概率），和为1
个人理解: 训练数据是一大堆的词对(cat,can) (can tree).  测试数据就是单个词 dog， 就会得到可能是dog上下文的词，
取概率最高的几个就行了(训练数据窗口大小)
```

```python
#skip-gram实现思路
#1.读取已经去除符号和分好词的数据
with open('data/Javasplittedwords22',encoding = 'utf-8') as f:
    text = f.read()
#2.去除低频词
words_count = Counter(words)
words = [w for w in words if words_count[w] > 50]

#3.构建所有词的int类型映射
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}

#4.对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]

#5.去除停用词
p = 1 - 根号（t / 频率） t = 1e-5

#6.构建网络
输入层 - embedding - negative sampling
```

word2vec有两种优化
========
1。Hieraechical softmax
------------

基于霍夫曼书，减少输出层计算量。计算量从v变成了logv

2。Negative sampling
-------
因为每一次训练样本经过模型都会更改权重，而权重数量太多了。 例如输入(fox, quick)，我们希望quick节点为1其他点为0，这些点就是negative word
使用负采样的时候，随即选择5-20个negative word更新权重，当然了quick这个点也会更新，因为和为1。
使用一元模型选择negative word 一个单词被选作negative sample的概率跟它出现的频次有关，出现频次越高的单词越容易被选作negative words。

3.高频词抽样
-------
对于训练原始文本中遇到的每一个单词，它们都有一定概率被我们从文本中删掉，而这个被删除的概率与单词的频率有关。

4.将常见词组和起来当一个词
---------

word2vec应用场景
---------
1.找出某个词的相近词集合
2.查看两个词相似程度
3.找出几个词中不同类的词

RNN and LSTM
-----------

![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-04-07%20at%2018.37.03.png)

```bash
相比RNN只有一个传递状态  h^t  ，LSTM有两个传输状态，一个  c^t （cell state），和一个  h^t （hidden state）。（Tips：RNN中的 h^t 对于LSTM中的 c^t ）

其中对于传递下去的 c^t 改变得很慢，通常输出的 c^t 是上一个状态传过来的 c^{t-1} 加上一些数值。

而 h^t 则在不同节点下往往会有很大的区别。
```
