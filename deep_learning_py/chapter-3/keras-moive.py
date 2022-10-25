from tensorflow.keras.datasets import imdb

# 获取数据集
# num_words=10000 的意思是仅保留训练数据中前10000个最常出现的单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# 数字标识单词索引
print("train data [0]:{}".format(train_data[0]))
# 表示用户评价 0：负面 1：正面
print(train_labels[0])

# 某条评论迅速解码为英文单词
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# 准备数据
import numpy as np


# 对列表进行 one-hot 编码，将其转换为 0 和 1 组成的向量。举个例子，序列 [3, 5] 将会
# 被转换为 10 000 维向量，只有索引为 3 和 5 的元素是 1，其余元素都是 0。然后网络第
# 一层可以用 Dense 层，它能够处理浮点数向量数据
# 将整数序列编码为二进制矩阵,1D张量
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(x_train[0])

# 构建网络,模型定义
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
#  Dense 层的参数（ 16）是该层隐藏单元的个数。一个隐藏单元（ hidden unit）是该层
# 表示空间的一个维度,输入数据投影到 16 维表示空间中（然后再加上偏置向量 b 并应用 relu 运算(非线性运算)
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
from tensorflow.keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 10000数据作为验证集
x_val = x_train[:10000]
# 10000数据作为训练集
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 现在使用 512 个样本组成的小批量，将模型训练 20 个轮次（即对 x_train 和 y_train 两
# 个张量中的所有样本进行 20 次迭代）。
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘制训练损失和验证损失
# 损失是一个数值，表示对于单个样本而言模型预测的准确程度
import matplotlib.pyplot as plt

history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
# 绘制训练精度和验证精度
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 修改4层,评估模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
