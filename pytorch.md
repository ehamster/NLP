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
