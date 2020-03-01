#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/3/1-22:25
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-basic
# @refere   :
# endregion

import torch

# 创建随机数tensor
x = torch.empty(5, 3)
print(x)
# 创建指定类型的tensor
y = torch.zeros(5, 3, dtype=torch.long)
print(y)
# 从已经存在的Tensor中创建新的Tensor
z = torch.tensor([5, 2, 3])
print(z)

print(x.new_ones(5, 3, dtype=torch.double))
# 改变了数据类型
print(torch.randn_like(x, dtype=torch.float))

# 加法
add1 = torch.rand(5, 3)
print(x + add1)

print(torch.add(add1, x))

z = torch.empty(5, 3)
torch.add(x, add1, out=z)
print(z)

print(add1.add_(x))

# 定位
print(x[:, 1])
x = torch.randn(4, 4)
print(x)
# 拉平
y = x.view(16)
print(y)
# 按照位置计算
print(x.view(-1, 8))
print(x.size(), y.size(), z.size())

# tensor转换层普通python
x = torch.randn(1)
print(x)
print(x.item())

# tensor转numpy
a = torch.ones(5)
print(a)
print(a.numpy())
import numpy as np

# numpy 转tensor
print(np.ones(5))
print(torch.from_numpy(np.ones(5)))

# 使用GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # cuda设备对象
    y = torch.ones_like(x, device=device)

    x = x.to(device)  # 后者使用x.to("cuda")
    z = x + y
    print(z)
print("cpu------>", z.to("cpu", torch.double))
