from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# 降低拟合
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# print(model.summary())

from tensorflow.keras import optimizers

# 样本（ sample）或输入（ input）：进入模型的数据点。
#  预测（ prediction）或输出（ output）：从模型出来的结果。
#  目标（ target）：真实值。对于外部数据源，理想情况下，模型应该能够预测出目标。
#  预测误差（ prediction error）或损失值（ loss value）：模型预测与目标之间的距离。
#  类别（ class）：分类问题中供选择的一组标签。例如，对猫狗图像进行分类时，“狗”
# 和“猫”就是两个类别。
#  标签（ label）：分类问题中类别标注的具体例子。比如，如果 1234 号图像被标注为
# 包含类别“狗”，那么“狗”就是 1234 号图像的标签。
#  真值（ ground-truth）或标注（ annotation）：数据集的所有目标，通常由人工收集。
#  二分类（ binary classification）：一种分类任务，每个输入样本都应被划分到两个互
# 斥的类别中。
#  多分类（ multiclass classification）：一种分类任务，每个输入样本都应被划分到两个
# 以上的类别中，比如手写数字分类。
#  多标签分类（ multilabel classification）：一种分类任务，每个输入样本都可以分配多
# 个标签。举个例子，如果一幅图像里可能既有猫又有狗，那么应该同时标注“猫”
# 标签和“狗”标签。每幅图像的标签个数通常是可变的。
#  标量回归（ scalar regression）：目标是连续标量值的任务。预测房价就是一个很好的
# 例子，不同的目标价格形成一个连续的空间。
#  向量回归（ vector regression）：目标是一组连续值（比如一个连续向量）的任务。如
# 果对多个值（比如图像边界框的坐标）进行回归，那就是向量回归。
#  小批量（ mini-batch）或批量（ batch）：模型同时处理的一小部分样本（样本数通常
# 为 8~128）。样本数通常取 2 的幂，这样便于 GPU 上的内存分配。训练时，小批量
# 用来为模型权重计算一次梯度下降更新。
# 二元交叉熵作为损失函数
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "cats_and_dogs_small/train"
validation_dir = "cats_and_dogs_small/validation"

# 将所有图像乘以 1/255 缩放
# train_datagen = ImageDataGenerator(rescale=1. / 255)
# 增强型数据生成器,实现图片随机变换，增加数据样本
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
# 拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

# 绘制训练过程中的损失曲线和精度曲线
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
