from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

model = VGG16(weights='imagenet')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# The local path to our target image
img_path = 'creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
# print(x.shape)

# 添加一个维度，将数组转换为
# (1, 224, 224, 3) 形状的批量,axis为需添加维度的轴,AXIS=0在原数组前添加维度
x = np.expand_dims(x, axis=0)
# print(x.shape)

# 对批量进行预处理（按通道进行颜色标准化），对颜色通道进行归一化
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

np.argmax(preds[0])

# 预测向量中的“非洲象”元素
african_elephant_output = model.output[:, 386]

# block5_conv3 层的输出特征图，
# 它是 VGG16 的最后一个卷积层
last_conv_layer = model.get_layer('block5_conv3')

# “非洲象”类别相对于 block5_conv3输出特征图的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# 访问刚刚定义的量：对于给定的样本图像，
# pooled_grads 和 block5_conv3 层的输出特征图
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# 对于两个大象的样本图像，这两个量都是 Numpy 数组
pooled_grads_value, conv_layer_output_value = iterate([x])

# 将特征图数组的每个通道乘以“这个通道对‘大象’类别的权重
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 得到的特征图的逐通道平均值即为类激活的热力图
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

K.clear_session

import cv2

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('elephant_cam.jpg', superimposed_img)
