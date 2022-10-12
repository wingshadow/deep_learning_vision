import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import face_recognition
# 找到人脸特征点
# 通过PIL加载图片
image = face_recognition.load_image_file("image/obama.jpg")

# 找到图像中所有人脸的所有面部特征，返回字典
face_landmarks_list = face_recognition.face_landmarks(image)

# 发现人脸数
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# 创建展示结果的图像
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

# 绘制关键点
for face_landmarks in face_landmarks_list:

    # 打印此图像中每个面部特征的位置
    # for facial_feature in face_landmarks.keys():
    # print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # 用一条线勾勒出图像中的每个面部特征
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

# jupyter 绘图
# pil_image.show()
plt.imshow(pil_image)
plt.axis('off')
plt.show()
