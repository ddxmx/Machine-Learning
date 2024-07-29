# Keras

## 一、样例模型的训练和预测

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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

