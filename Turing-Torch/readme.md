### torch basic 

##### 创建随机数tensor
```python
x = torch.empty(5, 3)
```

##### 创建指定类型的tensor
```python
y = torch.zeros(5, 3, dtype=torch.long)
```

##### 从已经存在的Tensor中创建新的Tensor
```python
z = torch.tensor([5, 2, 3])

```

##### 改变了数据类型
```python
torch.randn_like(x, dtype=torch.float)
```

##### 加法
```python
add1 = torch.rand(5, 3)
print(x + add1)

print(torch.add(add1, x))

z = torch.empty(5, 3)
torch.add(x, add1, out=z)
print(z)

print(add1.add_(x))
```

##### 定位
```python
print(x[:, 1])
x = torch.randn(4, 4)
```


##### 拉平
```python
y = x.view(16)
print(y)
```

##### 按照位置计算
```python
print(x.view(-1, 8))
print(x.size(), y.size(), z.size())
```


##### tensor转换层普通python
```python
x = torch.randn(1)
print(x)
print(x.item())
```


##### tensor转numpy
```python
a = torch.ones(5)
print(a)
print(a.numpy())
import numpy as np
```


##### numpy 转tensor
```python
print(np.ones(5))
print(torch.from_numpy(np.ones(5)))
```


##### 使用GPU
```python
if torch.cuda.is_available():
    device = torch.device("cuda")  # cuda设备对象
    y = torch.ones_like(x, device=device)

    x = x.to(device)  # 后者使用x.to("cuda")
    z = x + y
    print(z)
print("cpu------>", z.to("cpu", torch.double))
```
