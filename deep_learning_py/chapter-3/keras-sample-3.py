from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# keras工作流程
# 定义训练数据：输入张量和目标张量。
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))

# 同上述代码定义模型一样
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)
# 定义层组成的网络（或模型），将输入映射到目标。

# 配置学习过程：选择损失函数、优化器和需要监控的指标
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])
# 调用模型的 fit 方法在训练数据上进行迭代
model.fit(input_tensor, output_tensor, batch_size=128, epochs=10)
