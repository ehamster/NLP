1.常用语句
=======================
```bash
1.x = torch.randn(5,4)

2.y.add_(x)  相当于y = y + x
任何方法后面有_都意味着会改变前面的值

3.resize:
y = x.view(-1,2)

4.get value from one element tensor:
y[1,1].item()

5.numpy和tensor共享内存数值（假如在cpu）
改变一个，另一个也会变
a = torch.ones(5)
b = a.numpy()
a.add_(1)
b也变了，反过来也一样
c = torch.from_numpy(b)

6.tensor从cpu移动到gpu
device = torch.device("cuda")
x = x.to(device)
y = torch.ones_like(x,device=device)
z.to("cpu",torch.double)


7.
```
2.模型的保存读取
===================

```bash
A.保存模型的引用(常用)
只保存了模型的参数，一般保存为.pt或者.pth
1.save
torch.save(model.state_dict(),PATH)

state_dict()是一个dictionary
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])

2.load

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

B.保存整个模型  使用的是pickle module
1.save
torch.save(model,PATH)
2.Load
model = torch.load(PATH)
model.eval()

C.保存模型和其他东西用来再训练
1.save 一般保存为.tar
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
2.load    
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

D.读取不同模型
1.save
torch.save(modelA.state_dict(),PATH)

2.LOAD  设置strict =false就可以在多了层参数的key或者少了key的时候也能读取了
modelB = TheModelBClass(*args,**kwargs)
modelB.load_state_dict(torch.load(PATH),strict=False)
如果想用某一层的参数，但是层名字(key)不一样，在  state_dict里直接改key的名字就好了

E.Save on GPU load on CPU
1.save
torch.save(model.state_dict(), PATH)
2.load
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

F.Save on GPU, Load on GPU
1.save
torch.save(model.state_dict(), PATH)

2.load
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

G Saving torch.nn.DataParallel Models
1.save
torch.save(model.module.state_dict(), PATH)
2.load
# Load to whatever device you want
```
3.Torch Server
====================

```bash
1.先使用torch-model-archiver  将模型的ckpt和state_dict打包成一个.mar文件

```
