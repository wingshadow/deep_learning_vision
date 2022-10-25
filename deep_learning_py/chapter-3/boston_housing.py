from tensorflow.keras.datasets import boston_housing

# 波士顿房间预测
# 404 个训练样本和 102 个测试样本，每个样本都有 13 个数值特征，比如
# 人均犯罪率、每个住宅的平均房间数、高速公路可达性等
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

# 数据标准化
# 输入数据的每个特征（输入数据矩阵中的列），减去特征平均值，再除
# 以标准差，这样得到的特征平均值为 0，标准差为 1
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# 创建模型
from tensorflow.keras import models
from tensorflow.keras import layers


# 编译网络用的是 mse 损失函数,即均方误差（ MSE， mean squared error），预测值与目标值之差的平方
# 平均绝对误差（ MAE,mean absolute error.它是预测值与目标值之差的绝对值

def build_model():
    # 因为需要将同一个模型多次实例化,所以用一个函数来构建模型
    # 输入13特征值
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# 数据点很少，验证集会非常小（比如大约
# 100 个样本）。因此，验证分数可能会有很大波动，这取决于你所选择的验证集和训练集。也就
# 是说，验证集的划分方式可能会造成验证分数上有很大的方差,采用
# 使用 K 折交叉验证（见图 3-11）。这种方法将可用数据划分为 K
# 个分区（ K 通常取 4 或 5），实例化 K 个相同的模型，将每个模型在 K-1 个分区上训练，并在剩
# 下的一个分区上进行评估。模型的验证分数等于 K 个验证分数的平均值

# K 折验证
import numpy as np

k = 4
num_val_samples = len(train_data) // k
# num_epochs = 100
# all_scores = []
# for i in range(k):
#     print('processing fold #', i)
#     # 准备验证数据：第 k 个分区的数据
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#
#     # 准备训练数据：其他所有分区的数据
#     partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
#                                         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
#
#     # 构建 Keras 模型（已编译）
#     model = build_model()
#     # 训练模型（静默模式，verbose=0）
#     model.fit(partial_train_data, partial_train_targets,
#               epochs=num_epochs, batch_size=1, verbose=0)
#     # 在验证数据上评估模型
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
#     # 打印平均值
#     print(np.mean(all_scores))

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

# 计算所有轮次中的 K 折验证分数平均值
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 绘制验证分数
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 绘制验证分数（删除前 10 个数据点）
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 训练最终模型
model = build_model()
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score)
print(test_mae_score)
