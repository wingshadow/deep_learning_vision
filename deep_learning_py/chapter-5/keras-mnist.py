# 卷积神经网络
# 密集连接层和卷积层的根本区别在于， Dense 层从输入特征空间中学到的是全局模式，卷积层学到的是局部模式
# 卷积的工作原理：在 3D 输入特征图上滑动（ slide）这些 3× 3 或 5× 5 的窗口，在每个可能
# 的位置停止并提取周围特征的 3D 图块［形状为 (window_height, window_width, input_
# depth)］。然后每个 3D 图块与学到的同一个权重矩阵［叫作卷积核（ convolution kernel）］做
# 张量积，转换成形状为 (output_depth,) 的 1D 向量


from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
# Conv2D卷积
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 最大池化是从输入特征图中提取窗口，并输出每个通道的最大值
# 最大池化的作用：对特征图进行下采样
# 最大池化通常使用 2× 2 的窗口和步幅 2，其目
# 的是将特征图下采样 2 倍
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
