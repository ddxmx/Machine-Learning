# Keras

## 一、Keras概述

### 1、模型堆叠

 最常见的模型类型是层的堆叠：tf.keras.Sequential 模型 

~~~python
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
~~~

### 2、训练和评估

#### 2.1、设置训练流程

~~~python
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])
~~~

2.2、训练数据

2.2.1、使用numpy传入数据

~~~python
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

model.fit(train_x, train_y, epochs=10, batch_size=100,validation_data=(val_x, val_y))
~~~

#### 2.2、使用tf.data构造训练集

~~~python
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset, validation_steps=3)
~~~

#### 2.3、模型评估

~~~python
# 模型评估
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
model.evaluate(test_x, test_y, batch_size=32)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
~~~

#### 2.4、模型预测

~~~python
result = model.predict(test_x, batch_size=32)
print(result)
~~~

#### 2.5、回调

回调是传递给模型以自定义和扩展其在训练期间的行为的对象。我们可以编写自己的自定义回调，或使用tf.keras.callbacks中的内置函数，常用内置回调函数如下：

- tf.keras.callbacks.ModelCheckpoint：定期保存模型的检查点。
- tf.keras.callbacks.LearningRateScheduler：动态更改学习率。
- tf.keras.callbacks.EarlyStopping：验证性能停止提高时进行中断培训。
- tf.keras.callbacks.TensorBoard：使用TensorBoard监视模型的行为 。

~~~python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_x, train_y, batch_size=16, epochs=5,
         callbacks=callbacks, validation_data=(val_x, val_y))
~~~

### 3、模型保存和恢复

#### 3.1、权重保存

~~~python
# 权重保存与重载
model.save_weights('./weights/model')
model.load_weights('./weights/model')
# 保存为h5格式
model.save_weights('./model.h5', save_format='h5')
model.load_weights('./model.h5')
~~~

#### 3.2、网络结构保存

~~~python
# 序列化成json
import json
import pprint
'''
{'backend': 'tensorflow',
 'class_name': 'Sequential',
 'config': {'layers': [{'class_name': 'Dense',
                        'config': {'activation': 'relu',
                                   'activity_regularizer': None,
                                   'batch_input_shape': [None, 32],
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {}},
                                   'bias_regularizer': None,
                                   'dtype': 'float32',
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'GlorotUniform',
                                                          'config': {'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'dense_23',
                                   'trainable': True,
                                   'units': 64,
                                   'use_bias': True}},
                       {'class_name': 'Dense',
                        'config': {'activation': 'softmax',
                                   'activity_regularizer': None,
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {}},
                                   'bias_regularizer': None,
                                   'dtype': 'float32',
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'GlorotUniform',
                                                          'config': {'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'dense_24',
                                   'trainable': True,
                                   'units': 10,
                                   'use_bias': True}}],
            'name': 'sequential_6'},
 'keras_version': '2.2.4-tf'}
'''
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
# 从json中重建模型
fresh_model = tf.keras.models.model_from_json(json_str)
~~~

保存为yaml

~~~python
# 保持为yaml格式  #需要提前安装pyyaml

yaml_str = model.to_yaml()
'''
backend: tensorflow
class_name: Sequential
config:
  layers:
  - class_name: Dense
    config:
      activation: relu
      activity_regularizer: null
      batch_input_shape: !!python/tuple [null, 32]
      bias_constraint: null
      bias_initializer:
        class_name: Zeros
        config: {}
      bias_regularizer: null
      dtype: float32
      kernel_constraint: null
      kernel_initializer:
        class_name: GlorotUniform
        config: {seed: null}
      kernel_regularizer: null
      name: dense_23
      trainable: true
      units: 64
      use_bias: true
  - class_name: Dense
    config:
      activation: softmax
      activity_regularizer: null
      bias_constraint: null
      bias_initializer:
        class_name: Zeros
        config: {}
      bias_regularizer: null
      dtype: float32
      kernel_constraint: null
      kernel_initializer:
        class_name: GlorotUniform
        config: {seed: null}
      kernel_regularizer: null
      name: dense_24
      trainable: true
      units: 10
      use_bias: true
  name: sequential_6
keras_version: 2.2.4-tf

'''
print(yaml_str)
# 从yaml数据中重新构建模型
fresh_model = tf.keras.models.model_from_yaml(yaml_str)
~~~

#### 3.3、保存整个模型

~~~python
# 保存整个模型
model.save('all_model.h5')
# 导入整个模型
model = tf.keras.models.load_model('all_model.h5')
~~~

## 二、图像识别模型

~~~python
import tensorflow as tf
from tensorflow import keras

# 加载数据集，返回训练集和测试集
fashion_mnist = keras.datasets.fashion_mnist
'''
图像是28x28的NumPy数组，像素值介于0到255之间。
标签是整数数组，介于0到9之间。这些标签对应于图像所代表的服装类
'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#  训练集中有60,000个图像，每个图像由28 x 28的像素表
print(train_images.shape)  # (60000, 28, 28)
print(len(train_labels))  # 60000
print(test_images.shape)  # (10000, 28, 28)
print(len(test_labels))  # 10000

# 预处理数据，将值缩小到0到1之间
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建模型
model = keras.Sequential([
    # 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 第一个 Dense 层有 128 个节点（或神经元）
    tf.keras.layers.Dense(128, activation='relu'),
    # 第二个（也是最后一个）层会返回一个长度为 10 的 logits 数组
    tf.keras.layers.Dense(10)
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
'''
Epoch 1/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.4997 - accuracy: 0.8227
Epoch 2/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3768 - accuracy: 0.8629
Epoch 3/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3391 - accuracy: 0.8760
Epoch 4/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3153 - accuracy: 0.8829
Epoch 5/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2945 - accuracy: 0.8914
Epoch 6/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2807 - accuracy: 0.8968
Epoch 7/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2667 - accuracy: 0.9010
Epoch 8/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2564 - accuracy: 0.9043
Epoch 9/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2480 - accuracy: 0.9077
Epoch 10/10
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2379 - accuracy: 0.9106
'''
model.fit(train_images, train_labels, epochs=10)

# 评估准确率
'''
313/313 - 0s - loss: 0.3410 - accuracy: 0.8807 - 340ms/epoch - 1ms/step
Test accuracy: 0.8806999921798706
'''
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# 模型预测
probability_model = tf.keras.Sequential([model, keras.layers.Softmax()])
'''
313/313 [==============================] - 0s 723us/step
10000
tf.Tensor(9, shape=(), dtype=int64)
'''
predictions = probability_model.predict(test_images)
print(len(predictions))
print(tf.argmax(tf.constant(predictions[0])))

# 使用训练好的模型
img = test_images[1]
# 将单个样本设置成批样本，整体预测
img = tf.expand_dims(tf.constant(img), 0)
'''
1/1 [==============================] - 0s 41ms/step
tf.Tensor(2, shape=(), dtype=int64)
'''
predictions_single = probability_model.predict(img)
print(tf.argmax(tf.constant(predictions_single[0])))
~~~

