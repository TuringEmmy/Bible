

### task1:收集数据，训练网络模型

### task2:数据预处理，图像增强，网络结构
#### 数据预处理
是否亮度调整
是否归一化(Normalization)
是否需要将整个图像的内容，剪切图像的一部分可以吗
是否需要其它图像处理
#### 图像增强(解决训练数据不平衡(unbalanced data))
可以🌹一个方向盘角度分布直方图(histgoram),看看是否有些转动角度对应的体育系爱哪个的量远远大于其它
如果存在数据不平衡，如果不做任何处理，训练处理的效果如何
数据增强的方式
1. Under sampling(欠采样)
2. Over sampling（过采样）
3. 使用定制的损失函数，给予相对数量少的样本很大的权重
4. 收集已有的训练数据人工合成新的数据SMOTE:Synthetic Minority Oversampling Technique http://jair.org/index.php/jair/article/view/10302

#### 网络结构
根据end to end learning for self driving进行keras编程网络模型
使用训练的数据集和validation数据集上面的bias和variance来判断网络是否过拟合/欠拟合。然后调整网络结构

### task3:性能测试
如何测试训练好的神经网络
模拟器测试
通过损失函数判断
- 使用tensorboard 鸡洗净损失函数的查看

### task4:
