模型学习方法，会其意，知其形。先动他的目的的理念，再看公式
================
![](https://github.com/ehamster/NLP/blob/master/images/conclusion.jpg)




1.最大熵模型：
---------------
  分担风险，考虑所有可能，但是实现复杂效率低
  
  
2.HMM隐马尔科夫：
--------------

```bash
  我们使用一个隐马尔科夫模型（HMM）对这些例子建模。这个模型包含两组状态集合和三组概率集合：
　　* 隐藏状态：一个系统的（真实）状态，可以由一个马尔科夫过程进行描述（例如，天气）。
　　* 观察状态：在这个过程中‘可视’的状态（例如，海藻的湿度）。
　　* pi向量：包含了（隐）模型在时间t=1时一个特殊的隐藏状态的概率（初始概率）。
　　* 状态转移矩阵：包含了一个隐藏状态到另一个隐藏状态的概率
　　* 混淆矩阵：包含了给定隐马尔科夫模型的某一个特殊的隐藏状态，观察到的某个观察状态的概率。
　　因此一个隐马尔科夫模型是在一个标准的马尔科夫过程中引入一组观察状态，以及其与隐藏状态间的一些概率关系。
  
   一旦一个系统可以作为HMM被描述，就可以用来解决三个基本问题。其中前两个是模式识别的问题：给定HMM求一个观察序列的概率（评估）；
   搜索最有可能生成一个观察序列的隐藏状态序列（解码）。第三个问题是给定观察序列生成一个HMM（学习）。
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

4.Word2Vec
-----------------
```bash
他是word embedding的一种

如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『Skip-gram 模型』
而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『CBOW 模型』
eg  CBOW根据上下文求中心词
上下文的onehotencoding * w 就得到word2vec。  worc2vec * w‘ 得到中心词
比如，在skipgram来说：
  input: 1xV one hot encoding   假设一共V个词
  hidern layer : VxN W  N远远小于V
  output: 1XN * NxV W'  = 1XV
```
