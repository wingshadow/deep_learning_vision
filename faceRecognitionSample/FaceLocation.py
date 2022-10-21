# 2D绘图库
import base64
from io import BytesIO
from typing import re

import matplotlib.pyplot as plt
from PIL import Image
import face_recognition

# 找到或定位人脸
# 通过PIL加载图片

img = Image.open("image/o.jpg")
# width, height = img.size
# if width > 1600:
#     resize_img = img.resize((200, 260), Image.LANCZOS)
# buffered = BytesIO()
# resize_img.save(buffered, format="JPEG")
# img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
# byte_data = base64.b64decode(img_str)
# image_data = BytesIO(byte_data)

image = face_recognition.load_image_file("image/o.jpg")
if image.shape[1] > 1600:
    resize_img = img.resize((200, 260), Image.LANCZOS)
    buffered = BytesIO()
    resize_img.save(buffered, format="JPEG")
    image = face_recognition.load_image_file(buffered)

# 基于hog机器学习模型进行人脸识别，不能使用gpu加速
face_locations = face_recognition.face_locations(image)

# 找到几张人脸
print("I found {} face(s) in this photograph.".format(len(face_locations)))
for face_location in face_locations:
    # 打印人脸信息
    top, right, bottom, left = face_location
    print(
        "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # 提取人脸
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    # jupyter 绘图
    # pil_image.show()
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()
