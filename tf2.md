```bash
最大的区别是默认模式为动态模式
Eager execution
```
1.基本数据单位（Tensor）就是数组
=====================
```bash
    包含3属性，A.shape  A.dtype A.numpy()值
```
2.自动求导机制，使用 tf.GradientTape()
=======================
```python
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
```

3.Model and layer 
====================
```bash
model主要是重写这两个方法
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法

```
4.模型的训练 loss和optimizer
======================
```bash
 num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
``` 
5.模型的评估 tf.keras.metrics批预测
================
```bash
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
  ```
    
 6.模型的保存之   save_model
 ===================
 ```bash
 tf.saved_model.save(model,'/home/models/1/')
 保存了完整的模型，可以用来直接做推断，但是好像不能继续训练
 注意：如果是用上面继承keras.model的类构建了model
 1.call方法需要标记为静态图模式   @tf.function,  
 2.要主动调用call方法    y_pred = model.call(X)
 载入
 model=tf.saved_model.load('/home/models/1')
 ```
 7.注意！！！！
 =====================
 ```bash
 模型保存前最好用model.predict跑一下预测，初始化input，否则会报错
 ```
 8.embedding层
 ======================
 ```bash
 self.embedding_matrix = np.random.random([len_word_index,256])
 inputs = tf.nn.embedding_lookup(self.embedding_matrix,inputs)

```
9.关于@tf.function
==================
```bash
并不是任何函数都可以被 @tf.function 修饰！@tf.function 使用静态编译将函数内的代码转换成计算图，因此对函数内可使用的语句有一定限制（仅支持 Python 语言的一个子集），且需要函数内的操作本身能够被构建为计算图。建议在函数内只使用 TensorFlow 的原生操作，不要使用过于复杂的 Python 语句，函数参数只包括 TensorFlow 张量或 NumPy 数组，并最好是能够按照计算图的思想去构建函数（换言之，@tf.function 只是给了你一种更方便的写计算图的方法，而不是一颗能给任何函数加速的 银子弹 

在@tf.function的函数里要尽量避免用python自带的变量
函数会把支持转化为tf方法的方法变为计算图，下次来同样类型的tf值或者同值的python值只会复用计算图，不会把函数从新跑一遍
对于 Python 内置的整数和浮点数类型，只有当值完全一致的时候， @tf.function 才会复用之前建立的计算图，而并不会自动将 Python 内置的整数或浮点数等转换成张量。
要使用tf.constant(1,dtype=tf.int32)
对于这种tf的值，类型一样的话就会复用计算图
```

```python
@tf.function
def f(x):
    print("The function is running in Python")
    tf.print(x)

a = tf.constant(1, dtype=tf.int32)
f(a)
b = tf.constant(2, dtype=tf.int32)
f(b)
b_ = np.array(2, dtype=np.int32)
f(b_)
c = tf.constant(0.1, dtype=tf.float32)
f(c)
d = tf.constant(0.2, dtype=tf.float32)
f(d)


The function is running in Python
1
2
2
The function is running in Python
0.1
0.2
print()不支持转为计算图,只支持tf.print
```
10.checkpoint
==================
```bash

# train.py 模型训练阶段
model = MyModel()
# 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
checkpoint = tf.train.Checkpoint(myModel=model)
# ...（模型训练代码）
# 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
checkpoint.save('./save/model.ckpt')

# test.py 模型使用阶段

model = MyModel()
checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
# 模型使用代码
```
