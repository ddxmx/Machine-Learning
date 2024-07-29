# TensorFlow

# 一、TensorFlow基本使用

## 1、基本数据类型

#### 1.1、数值类型

标量：单个实数，如1.2，维度为0，shape为[]

向量：包含n个元素的有序集合，如[1.2,3.4]，维度为1，shape为[n]

矩阵：m行n列的有序集合，如[[1,2],[3,4]]，维度为2，shape为[m,n]

张量：维度>2的数组都称为张量。在 TensorFlow 中，一般把标量、向量、矩阵也统称为张量，不作区分

张量常用的精度类型有 tf.int16、**tf.int32、tf.int64**、tf.float16、**tf.float32、 tf.float64** 等，其中 tf.float64 即为 tf.double

~~~python
# 标量
a = 1.2
aa = tf.constant(a)
print(type(a))  # <class 'float'>
print(type(aa))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
print(tf.is_tensor(aa))  # True
~~~

~~~python
# 向量
x = tf.constant([1, 2., 3.3])
print(x)  # tf.Tensor([1.  2.  3.3], shape=(3,), dtype=float32)
# 将tf张量的数据导出为numpy数组格式
print(x.numpy())  # [1.  2.  3.3]
# 张量的数值精度
print(x.dtype)  # <dtype: 'float32'>
# 定义时可以指定数值精度
x = tf.constant([1, 2., 3.3], dtype=tf.float64)
print(x.dtype)  # <dtype: 'float64'>
~~~

```python
# 矩阵
'''
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
'''
x = tf.constant([[1, 2], [3, 4]])
print(x) 
```

**定义张量时，可以指定shape**

```python
'''
tf.Tensor(
[[1 2 3]
 [4 5 6]], shape=(2, 3), dtype=int32)
'''
x = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print(x)
```

#### 1.2、字符串类型

~~~python
x = tf.constant("hello")
print(x)  # tf.Tensor(b'hello', shape=(), dtype=string)
~~~

#### 1.3、布尔类型

~~~python
x = tf.constant([True, False])
print(x)  # tf.Tensor([ True False], shape=(2,), dtype=bool)
~~~

### 2、类型转换

**高精度向低精度转换可以会损失精度**

~~~python
x = tf.constant(1.2)
print(x)  # tf.Tensor(1.2, shape=(), dtype=float32)
x = tf.cast(x, tf.int32)
print(x)  # tf.Tensor(1, shape=(), dtype=int32)
~~~

**布尔类型和整型之间也可以进行转换，True转换为1，False转换为0**

~~~python
x = tf.constant([True, False])
x = tf.cast(x, tf.int32)
print(x)  # tf.Tensor([1 0], shape=(2,), dtype=int32)
~~~

**将非 0 数字都视为 True**

~~~python
x = tf.constant([2, 0, 1])
x = tf.cast(x, tf.bool)
print(x)  # tf.Tensor([ True False  True], shape=(3,), dtype=bool)
~~~

### 3、可变张量

使用tf.constant定义的张量都不可变，tf的操作将返回一个新的tf常量，使用tf.Variable可以定义变量

~~~python
a = tf.constant([1, 2, 3])
aa = tf.Variable(a)
print(aa)  # <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([1, 2, 3])>
print(tf.is_tensor(aa))  # True
# 可训练的
print(aa.trainable)  # True
~~~

### 4、创建张量

tf.constant()和 tf.convert_to_tensor()都能够把Numpy数组或者Python列表数据类型转化为Tensor类型

#### 4.1、从列表创建

~~~python
import tensorflow as tf
import numpy as np

x = tf.convert_to_tensor([1, 2.2])
print(x)  # tf.Tensor([1.  2.2], shape=(2,), dtype=float32)
~~~

#### 4.2、从numpy数组创建

**numpy的浮点型默认是float64位精度**

~~~python
'''
tf.Tensor(
[[1.2 2.3]
 [2.4 1.6]], shape=(2, 2), dtype=float64)
'''
x = np.array([[1.2, 2.3], [2.4, 1.6]])
x = tf.convert_to_tensor(x)
print(x)
~~~

#### 4.3、创建全为0或1的张量

~~~python
'''
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
'''
# 创建全为0的tensor
x = tf.zeros([2, 3])
print(x)

# 创建全为1的tensor
y = tf.ones([1, 2])
print(y)  # tf.Tensor([[1. 1.]], shape=(1, 2), dtype=float32)

# 创建全为0的tensor，shape和指定tensor一致，等价tf.zeros(x.shape)
x1 = tf.zeros_like(x)
print(x1)

'''
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
'''
# 创建全为1的tensor，shape和指定tensor一致，等价tf.ones(y.shape)
y1 = tf.ones_like(y)
print(y1)  # tf.Tensor([[1. 1.]], shape=(1, 2), dtype=float32)
~~~

**通过tf.fill函数可以自定义数值张量**

~~~python
'''
tf.Tensor(
[[5 5 5]
 [5 5 5]], shape=(2, 3), dtype=int32)
'''
x = tf.fill([2, 3], 5)
print(x)
~~~

#### 4.4、序列创建

~~~python
# 0~5的序列
x = tf.range(5)
print(x)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)
# 3~10的序列，步长为2
x = tf.range(3, 10, 2)
print(x)  # tf.Tensor([3 5 7 9], shape=(4,), dtype=int32)
~~~

#### 4.5、常用的随机分布张量

（1）正态分布

均值在一个正负标准差范围内数据占68%，两个正负标准差范围内数据站95%，三个正负标准差范围内数据占99.7% 

~~~python
'''
tf.Tensor(
[[ 8.7993355  5.709466  11.811717 ]
 [ 4.9151816  6.4586782  7.1852446]], shape=(2, 3), dtype=float32)
'''
# 默认均值是0，标准差是1.0
x = tf.random.normal([2, 3], mean=10, stddev=2.0)
print(x)
~~~

（2）截断的正态分布

**随机值在均值附近2个方差内**

~~~python
'''
tf.Tensor(
[[10.292586 11.381808  8.475887]
 [11.196225 10.889628 11.959349]], shape=(2, 3), dtype=float32)
'''
# 默认均值是0，方差是1.0
x = tf.random.truncated_normal([2, 3], mean=10, stddev=2.0)
print(x)
~~~

（3）均匀分布

~~~python
'''
tf.Tensor(
[[17 17 68 68]
 [77 16 73 57]
 [35 42 54 63]], shape=(3, 4), dtype=int32)
'''
x = tf.random.uniform([3, 4], minval=10, maxval=100, dtype=tf.int32)
print(x)
~~~

（4）泊松分布

泊松分布就是描述某段时间内，事件具体的发生概率。

λ是泊松分布所依赖的唯一参数。λ值愈小，分布愈偏倚，随着λ的增大，分布趋于对称。

当λ = 20时，分布泊松接近于正态分布。

~~~python
'''
tf.Tensor(
[[3. 4. 3. 4.]
 [5. 4. 2. 4.]
 [3. 6. 9. 2.]], shape=(3, 4), dtype=float32)
'''
x = tf.random.poisson(lam=5.0, shape=[3, 4])
print(x)
~~~

### 5、索引和切片

#### 5.1、 标准索引方式提取张量

标准索引方式，所在维度信息不会保留

~~~python
x = tf.random.normal([4, 32, 32, 3])
print(x[0].shape)  # (32, 32, 3)
print(x[0][1].shape)  # (32, 3)
print(x[0][1][2].shape)  # (3,)
print(x[2][1][0][1].shape)  # ()
~~~

#### 5.2、切片方式提取张量

切片方式，所在维度信息会保留

~~~python
x = tf.random.normal([4, 32, 32, 3])
# 第0个维度获取2个样本
print(x[1:3].shape)  # (2, 32, 32, 3)
# 等价于x[0].shape
print(x[0, ::].shape)  # (32, 32, 3)
# 切片时指定步长
print(x[:, 0:28:2, 0:28:2, :].shape)  # (4, 14, 14, 3)
~~~

**采样时可以设置步长**

~~~python
x = tf.range(9)
# 步长为-1时，表示倒序采样
print(x[5:2:-1].numpy())  # [5 4 3]
# 倒序间隔采样
print(x[::-2].numpy())  # [8 6 4 2 0]
~~~

**多个:可以使用...替代**

~~~python
# 采样1个通道数据
x = tf.random.normal([4, 32, 32, 3])
print(x[:, :, :, 2].shape)  # (4, 32, 32)
# 将多个:使用...替代
print(x[..., 2].shape)  # (4, 32, 32)
print(x[1:3, ..., 2].shape)  # (2, 32, 32)
~~~

#### 5.3、tf.gather根据索引号收集数据

 tf.gather 用于沿着张量的一个轴（维度）收集元素。它接受一个张量和一个索引张量作为输入，并根据索引张量中的索引来选择元素。 

~~~python
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
x1 = tf.gather(x, [0, 1], axis=0)  # 在班级维度收集第 1~2 号班级成绩册
print(x1.shape)  # (2, 35, 8)
x1 = tf.gather(x, [0, 3, 8, 11, 12, 26], axis=1)  # 收集第 1,4,9,12,13,27 号同学成绩
print(x1.shape)  # (4, 6, 8)
~~~

~~~python
# 如果希望抽查第[2,3]班级的第[3,4,6,27]号同学的科目成绩，则可以通过组合多个tf.gather实现。
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
students = tf.gather(x, [1, 2], axis=0)  # 收集第2,3号班级，shape:[2,35,8]
score = tf.gather(students, [2, 3, 5, 26], axis=1)  # 收集第3,4,6,27号同学
print(score.shape)  # (2, 4, 8)
~~~

#### 5.4、tf.gather_nd根据多维索引收集数据

 tf.gather_nd 用于基于多维索引来收集元素。它可以沿多个维度收集元素，而不仅仅是一个维度。tf.gather_nd 接受一个索引张量，其中的每一项是一个多维索引，指向源张量中的元素位置。 

~~~python
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
# 获取第2个班级第2个同学所有科目，获取第3个班级第3个同学所有科目，获取第4个班级第4个同学所有科目
x1 = tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]])
print(x1.shape)  # (3, 8)
x1 = tf.gather_nd(x, [[1, 1, 2], [2, 2, 3], [3, 3, 4]])
print(x1.shape)  # (3,)
~~~

#### 5.5、掩码收集数据

tf.boolean_mask 既可以实现了 tf.gather 方式的一维掩码采样，又可以实现 tf.gather_nd 方式的多维掩码采样。 

一维掩码采样，掩码的长度必须与对应维度的长度一致。

~~~python
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
# 指定维度掩码采样
x1 = tf.boolean_mask(x, mask=[True, False, False, True], axis=0)  # 在第一个维度进行掩码
print(x1.shape)  # (2, 35, 8)

# 多维掩码采样
x = tf.random.uniform([2, 3], maxval=100, dtype=tf.int32)  # 成绩册张量
x1 = tf.boolean_mask(x, [[True, True, False], [False, True, True]])
print(x1.shape)  # (4,)
~~~

#### 5.6、tf.where条件判断

通过 tf.where(cond, a, b)操作可以根据 cond 条件的真假从参数𝑨或𝑩中读取数据

~~~python
a = tf.fill([3, 3], 10)
b = tf.fill([3, 3], 5)
# 构造采样条件
cond = tf.constant([[True, False, False], [False, True, False], [True, True, False]])
x = tf.where(cond, a, b)  # 根据条件从a,b中采样
'''
tf.Tensor(
[[10  5  5]
 [ 5 10  5]
 [10 10  5]], shape=(3, 3), dtype=int32)
'''
print(x)

# a和b参数不指定，tf.where会返回cond张量中所有True的元素的索引坐标
x = tf.where(cond)
'''
tf.Tensor(
[[0 0]
 [1 1]
 [2 0]
 [2 1]], shape=(4, 2), dtype=int64)
'''
print(x)
~~~

如下为tf.where函数使用场景

~~~python
'''
tf.Tensor(
[[-0.7976382  -0.01695255 -1.2538124 ]
 [ 1.6824288   0.46764106  0.980897  ]
 [ 1.4840144   0.1947775  -1.4973005 ]], shape=(3, 3), dtype=float32)
'''
x = tf.random.normal([3, 3])
print(x)
mask = x > 0  # 比较操作，等同于 tf.math.greater()
'''
tf.Tensor(
[[False False False]
 [ True  True  True]
 [ True  True False]], shape=(3, 3), dtype=bool)
'''
print(mask)
# 恢复出所有正数的元素
x = tf.boolean_mask(x, mask)
print(x)  # tf.Tensor([1.6824288  0.46764106 0.980897   1.4840144  0.1947775 ], shape=(5,), dtype=float32)
~~~

#### 5.7、scatter_nd刷新张量数据

通过 tf.scatter_nd(indices, updates, shape)可以高效地刷新张量的部分数据，但是只能在全0张量上刷新 。

~~~python
# 创建一个3x3的全零张量
initial_tensor = tf.zeros([3, 3])
# 更新的索引和值
indices = tf.constant([[0, 1], [1, 2]])
updates = tf.constant([9.0, 5.0])
# 使用tf.scatter_nd进行更新
updated_tensor = tf.tensor_scatter_nd_update(initial_tensor, indices, updates)
'''
tf.Tensor(
[[0. 9. 0.]
 [0. 0. 5.]
 [0. 0. 0.]], shape=(3, 3), dtype=float32)
'''
print(updated_tensor)
~~~

### 6、维度变换

#### 6.1、改变视图

改变视图并不会改变张量的存储方式

~~~python
x = tf.range(96)
# 改变张量的视图方式为[2,4,4,3]
x = tf.reshape(x, [2, 4, 4, 3])
print(x.shape)  # (2, 4, 4, 3)
~~~

#### 6.2、增加维度

增加维度并不会改变张量的存储方式

在指定轴axis位置增加维度

~~~python
x = tf.random.uniform([28, 28])
# [28,28]在1的轴的位置插入维度[28,28,1]
x1 = tf.expand_dims(x, axis=2)
print(x1.shape)  # (28, 28, 1)
x2 = tf.expand_dims(x, axis=-2)
print(x2.shape)  # (28, 1, 28)
~~~

#### 6.3、复制维度

~~~python
'''
tf.Tensor(
[[1 2]
 [1 2]], shape=(2, 2), dtype=int32)
'''
b = tf.constant([1, 2])  # 维度为[2]
b = tf.expand_dims(b, axis=0)  # 维度增加为[1,2]
# 表示在0维复制成2倍，1维复制成1倍
b = tf.tile(b, multiples=[2, 1])
print(b)
~~~

#### 6.4、删除维度

删除维度并不会改变张量的存储方式

删除维度只能删除长度为1的维度

~~~python
# 删除指定维度索引位置的维度
x = tf.random.uniform([1, 28, 28, 1])
x1 = tf.squeeze(x, axis=3)
print(x1.shape)  # (1, 28, 28)
# 未指定维度时，表示删除所有为1的维度
x2 = tf.squeeze(x)
print(x2.shape)  # (28, 28)
~~~

#### 6.5、交换维度

**交换维度会改变张量的存储方式**

~~~python
x = tf.random.normal([2, 32, 32, 3])
# 将每个维度的索引编号[0,1,2,3]重新调整[0,3,1,2]
x = tf.transpose(x, perm=[0, 3, 1, 2])
print(x.shape)  # (2, 3, 32, 32)
~~~

### 7、Boradcasting机制

Broadcasting 称为广播机制(或自动扩展机制)，Broadcasting 机制都能通过优化手段避免实际复制数据而完成逻辑运算。

~~~python
x = tf.constant([[1, 2], [2, 3]])  # shape:[2,2]
b = tf.constant([5])  # shape:[1]

# 手动增加维度后，复制维度
b = tf.expand_dims(b, axis=1)
print(b)  # tf.Tensor([[5]], shape=(1, 1), dtype=int32)
b = tf.tile(b, [2, 2])
'''
tf.Tensor(
[[5 5]
 [5 5]], shape=(2, 2), dtype=int32)
'''
print(b)
~~~

~~~python
# 主动调用广播函数
x = tf.constant([[1, 2], [2, 3]])  # shape:[2,2]
b = tf.constant([5])  # shape:[1]

b = tf.broadcast_to(b, [2, 2])
'''
tf.Tensor(
[[5 5]
 [5 5]], shape=(2, 2), dtype=int32)
'''
print(b)
~~~

**张量进行自动进行广播的条件：先将张量shape靠右对齐，长度为1的维度可以自动扩展，不存在的维度，将增加维度后再进行自动扩展。**

如下运算会自动调用Broadcasting机制：+，-，*，/，矩阵乘法

~~~python
x = tf.constant([[1, 2, 3, 4], [11, 22, 33, 44]])  # shape:[2,4]
b = tf.constant([10, 20])  # shape:[2]
# 报错，无法自动进行广播，x的最右维度是4，b的最右维度是2，不相等，无法自动扩展
print(x + b)

# 手动复制后，最右维度一致后，可以自动扩展
b = tf.tile(b, [2])  # shape:[4]
'''
tf.Tensor(
[[11 22 13 24]
 [21 42 43 64]], shape=(2, 4), dtype=int32)
'''
print(x + b)
~~~

### 8、数学运算

#### 8.1、加减乘除

~~~python
a = tf.range(5)  # shape:[5]，[0,1,2,3,4]
b = tf.constant(2)  # shape:[1]，扩展成[2,2,2,2,2]
# 加法
print(a + b)  # tf.Tensor([2 3 4 5 6], shape=(5,), dtype=int32)
# 减法
print(a - b)  # tf.Tensor([-2 -1  0  1  2], shape=(5,), dtype=int32)
# 乘法
print(a * b)  # tf.Tensor([0 2 4 6 8], shape=(5,), dtype=int32)
# 除法
print(a / b)  # tf.Tensor([0.  0.5 1.  1.5 2. ], shape=(5,), dtype=float64)
# 整除
print(a // b)  # tf.Tensor([0 0 1 1 2], shape=(5,), dtype=int32)
# 取模
print(a % b)  # tf.Tensor([0 1 0 1 0], shape=(5,), dtype=int32)
~~~

#### 8.2、乘方运算

~~~python
x = tf.range(1, 5)
# 乘方：x的n次方
print(tf.pow(x, 3))  # tf.Tensor([ 1  8 27 64], shape=(4,), dtype=int32)
# 幂乘
print(x ** 3)  # tf.Tensor([ 1  8 27 64], shape=(4,), dtype=int32)
# 平法square
print(tf.square(x))  # tf.Tensor([ 1  4  9 16], shape=(4,), dtype=int32)
# 平方根square root，开方前需要转换为小数
x = tf.cast(x, tf.float32)
print(tf.sqrt(x))  # tf.Tensor([1.        1.4142135 1.7320508 2.       ], shape=(4,), dtype=float32)
~~~

#### 8.3、指数和对数

指数运算就是乘方运算，对于以自然指数e，tf有单独的支持

~~~python
x = tf.constant([1, 2, 3])
# 指数运算，e的x次方
x = tf.cast(x, tf.float32)
print(tf.exp(x))  # tf.Tensor([ 2.7182817  7.389056  20.085537 ], shape=(3,), dtype=float32)
# 对数运算，logeX
print(tf.math.log(x))  # tf.Tensor([0.        0.6931472 1.0986123], shape=(3,), dtype=float32)
~~~

#### 8.4、矩阵乘法

TensorFlow 中的 矩阵相乘可以使用批量方式，也就是张量𝑨和𝑩的维度数可以大于 2。**当张量𝑨和𝑩维度数大 于 2 时，TensorFlow 会选择𝑨和𝑩的最后两个维度进行矩阵相乘，前面所有的维度都视作 Batch 维度**。

**A和B能够矩阵乘法的条件：A的最后一个维度和B的倒数第二个维度长度必须相等。**

~~~python
a = tf.random.normal([4, 3, 28, 32])
b = tf.random.normal([4, 3, 32, 2])
# 最后2位进行矩阵乘法，其他位值必须一致，a的最后1位32和b的倒数第2位32必须一致
print(a @ b)
print(tf.matmul(a, b))
~~~

矩阵乘法支持自动广播机制

~~~python
a = tf.random.normal([4, 28, 32])
b = tf.random.normal([32, 16])
# 矩阵乘法支持自动广播机制，b会自动扩展为公共shape:[4,32,16]
print(tf.matmul(a, b).shape)  # (4, 28, 16)
~~~

### 9、合并和分割

#### 9.1、合并

合并是指将多个张量在某个维度上合并为一个张量。张量的合并可以使用拼接(Concatenate)和堆叠(Stack)操作实现。

拼接操作并不会产生新的维度，仅在现有的维度上合并，而堆叠会创建新维度。

**tf.concat拼接合并操作可以在任意的维度上进行，唯一的约束是非合并维度的长度必须一致。**

~~~python
a = tf.random.normal([4, 35, 8])  # 模拟成绩册 A
b = tf.random.normal([6, 35, 8])  # 模拟成绩册 B
x = tf.concat([a, b], axis=0)  # 拼接合并成绩册
print(x.shape)  # (10, 35, 8)
~~~

**tf.stack 也需要满足张量堆叠合并条件，它需要所有待合并的张量 shape 完全一致才可合并。**

~~~python
a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
x = tf.stack([a, b], axis=0)
print(x.shape)  # (2, 35, 8)
~~~

#### 9.2、分割 

tf.unstack可以在某个维度上，按照长度为1的方式分割，**分割的维度会消失**

~~~python
x = tf.random.normal([2, 3], mean=10, stddev=3.0)
'''
tf.Tensor(
[[11.273815  11.70267    2.680502 ]
 [ 9.2411375 15.929986  10.921686 ]], shape=(2, 3), dtype=float32)
'''
print(x)
# 返回的是张量的列表
x = tf.unstack(x, axis=0)
'''
[<tf.Tensor: shape=(3,), dtype=float32, numpy=array([11.273815, 11.70267 ,  2.680502], dtype=float32)>, 
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 9.2411375, 15.929986 , 10.921686 ], dtype=float32)>]
'''
print(x)
print(x[0])  # tf.Tensor([11.273815 11.70267   2.680502], shape=(3,), dtype=float32)
~~~

 tf.split可以指定分割的方案，当num_or_size_splits为单个数值时，如10表 示等长切割为 10 份；当 num_or_size_splits 为 List 时，List 的每个元素表示每份的长度，如[2,4,2,2]表示切割为4 份，每份的长度依次是 2、4、2、2。

~~~python
x = tf.random.normal([10, 35, 8])
# 指定分割的份数
result = tf.split(x, num_or_size_splits=10, axis=0)
print(len(result))  # 10
print(result[0].shape)  # (1, 35, 8)
# 指定分割的每份的大小
result = tf.split(x, num_or_size_splits=[4, 2, 2, 2], axis=0)
print(len(result))  # 4
print(result[1].shape)  # (2, 35, 8)
~~~

### 10、数据统计

#### 10.1、向量范数

向量范数是表征向量“长度”的一种度量方法，它可以推广到张量上。 在神经网络中，常用来表示张量的权值大小，梯度大小等。

~~~python
x = tf.ones([2, 2])
# 第1范数，向量𝒙的所有元素绝对值之和
x1 = tf.norm(x, ord=1)
print(x1)  # tf.Tensor(4.0, shape=(), dtype=float32)

# 第2范数，向量𝒙的所有元素的平方和，再开根号
x2 = tf.norm(x, ord=2)
print(x2)  # tf.Tensor(2.0, shape=(), dtype=float32)
~~~

#### 10.2、最大值、最小值、平均值、总和、最大值索引、最小值索引

当不指定 axis 参数时，tf.reduce_*函数会求解出全局元素的最大、最小、均值、和等 数据

~~~python
x = tf.constant([[2, 7, 3], [1, 9, 4]])
print(tf.reduce_max(x, axis=1))  # tf.Tensor([7 9], shape=(2,), dtype=int32)
print(tf.reduce_min(x))  # tf.Tensor(1, shape=(), dtype=int32)
print(tf.reduce_mean(x))  # tf.Tensor(4, shape=(), dtype=int32)
print(tf.reduce_sum(x))  # tf.Tensor(26, shape=(), dtype=int32)
# 计算第1个维度上最大值索引
print(tf.argmax(x, axis=1))  # tf.Tensor([1 1], shape=(2,), dtype=int64)
# 计算第0个维度上最小值索引
print(tf.argmin(x, axis=0))  # tf.Tensor([1 0 0], shape=(3,), dtype=int64)
~~~

#### 10.3、张量比较

tf.equal(a, b)(或 tf.math.equal(a,  b)，两者等价)函数可以比较2个张量是否相等

~~~python
out = tf.random.normal([100, 10])  # 定义100个样本，每个样本10个值
'''
tf.Tensor(
[4 7 8 1 0 3 5 6 6 5 5 7 4 0 2 6 9 8 1 6 2 9 3 7 9 1 5 4 9 4 6 9 1 0 2 1 2
 3 6 6 7 8 9 5 4 8 2 0 8 3 6 0 8 2 1 1 8 5 1 9 1 5 6 2 6 4 0 0 2 7 8 4 6 7
 0 5 2 8 8 7 8 7 1 2 8 0 1 8 0 4 3 3 7 6 5 3 5 0 7 7], shape=(100,), dtype=int64)
'''
out = tf.nn.softmax(out, axis=1)  # 每个样本值转换为概率
print(out)
pred = tf.argmax(out, axis=1)  # 计算预测值，获取每个样本的预期值索引
print(pred)
y = tf.random.uniform([100], dtype=tf.int64, maxval=10)  # 模型生成真实标签
out = tf.equal(pred, y)  # 预测值与真实值比较，返回布尔类型的张量
print(out)
out = tf.cast(out, dtype=tf.int32)  # 布尔型转int型
'''
tf.Tensor(
[0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0], shape=(100,), dtype=int32)
'''
print(out)
correct = tf.reduce_sum(out)  # 统计True的个数，False的值为0
print(correct)  # tf.Tensor(11, shape=(), dtype=int32)
accuracy = correct / tf.size(out)
print(accuracy)  # tf.Tensor(0.11, shape=(), dtype=float64)
~~~

### 11、填充

复制维度的方式可以进行填充，但是会破坏原有数据结构。通常的做法是，在需要补充长度的数据开始或结束处填充足够数量的特定数值，这些特定数值一般代表了无效意义，例如0。

填充操作可以通过tf.pad(x, paddings)函数实现，参数paddings是包含了多个
[Left Padding,Right Padding]的嵌套方案 List，如[[0,0],[2,1],[1,2]]表示第一个维度不填充，第二个维度左边(起始处)填充两个单元，右边(结束处)填充一个单元，第三个维度左边填充一个单元，右边填充两个单元。

~~~python
a = tf.constant([1, 2, 3, 4, 5, 6])  # 第一个句子
b = tf.constant([7, 8, 1, 6])  # 第二个句子
b = tf.pad(b, [[0, 2]])  # 句子末尾填充 2 个 0
print(b)  # tf.Tensor([7 8 1 6 0 0], shape=(6,), dtype=int32)
~~~

### 12、数据限幅

 数据限幅指的是对张量中元素的范围进行最大最小限制。‌ 

可以通过 tf.maximum(x, a)实现数据的下限幅，即𝑥 ∈ [𝑎, +∞)；可以通过 tf.minimum(x, a)实现数据的上限幅，即𝑥 ∈ (−∞,𝑎]。

~~~python
x = tf.range(9)
# 设置下限幅
print(tf.maximum(x, 2))  # tf.Tensor([2 2 2 3 4 5 6 7 8], shape=(9,), dtype=int32)
# 设置上限幅
print(tf.minimum(x, 7))  # tf.Tensor([0 1 2 3 4 5 6 7 7], shape=(9,), dtype=int32)
# 同时设置上下限幅
print(tf.clip_by_value(x, 2, 7))  # tf.Tensor([2 2 2 3 4 5 6 7 7], shape=(9,), dtype=int32)
~~~



