中文无字典分词方案-基于互信息和左右熵
==================
点间互信息
---------------------------------
![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-03-12%20at%2014.53.08.png)


左右信息熵
---------------------------------
![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-03-12%20at%2014.45.06.png)

之后就可以计算一个word的得分了
---------------------------------
![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-03-12%20at%2014.54.52.png)


不需要使用ngram的方法准个词计算，可以从左向右逐字添加的方法先计算互信息，得出满足条件的候选词，再算信息熵
![](https://github.com/ehamster/NLP/blob/master/images/Screenshot%202019-03-12%20at%2014.59.24.png)

代码(使用ngram的原始方法，效率慢):
```python
from nltk.probability import FreqDist  
f=open(r"C:\Users\user\Documents\Downloads\santi.txt")
text=f.read()
 
stop_word=['【','】',')','(','、','，','“','”','。','\n','《','》',' ','-','！','？','.','\'','[',']','：','/','.','"','\u3000','’','．',',','…','?']
for i in stop_word:
    text=text.replace(i,"")
    
min_entropy=0.8 
min_p=7
max_gram=4
count_appear=20
 
 
def gram(text,max_gram):
    t1=[i for i in text]
    loop=len(t1)+1-max_gram
    t=[]
    for i in range(loop):
        t.append(text[i:i+max_gram])
    if max_gram==1:
        return t1
    else:
        return t
import math
def pro(word):
    len_word=len(word)
    total_count=len(word_all[len_word])
    pro=freq_all[len_word][word]/total_count
    return pro
def entropy(alist):
    f=FreqDist(alist)
    ent=(-1)*sum([i/len(alist)*math.log(i/len(alist)) for i in f.values()])
    return ent
 
freq_all=[0]
word_all=[0]
for i in range(1,max_gram+1):
    t=gram(text,i)
    freq=FreqDist(t)
    word_all.append(t)
    freq_all.append(freq)
    
#筛选一部分符合互信息的单词
final_word=[]
for i in range(2,max_gram+1):
    for j in word_all[i]:
        if freq_all[i][j]<count_appear:
            pass
        else:
            p=min([pro(j[:i])*pro(j[i:]) for i in range(1,len(j))])
            if math.log(pro(j)/p)>min_p:
               final_word.append(j) 
final_word=list(set(final_word))   
#筛选左右熵
import re
final_word2=[]
for i in final_word:
    lr=re.findall('(.)%s(.)'%i,text)
    left_entropy=entropy([w[0] for w in lr])
    right_entropy=entropy([w[1] for w in lr])
    if min([right_entropy,left_entropy])>min_entropy:
        final_word2.append(i)
```
