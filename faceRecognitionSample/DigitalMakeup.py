import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import face_recognition
# 人脸特征点填充颜色
# 通过PIL加载图片
image = face_recognition.load_image_file("image/obama.jpg")

# 找到图像中所有人脸的所有面部特征，返回字典
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)

# 绘图
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # 眉毛涂色 fill是RGBA模式,RGBA代表红(Red)、绿(Green)、蓝(Blue)和Alpha通道(Alpha)。Alpha通道为图像的不透明度参数
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # 嘴唇涂色
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # 眼睛涂色
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # 眼线涂色
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

# jupyter 绘图
# pil_image.show()
plt.imshow(pil_image)
plt.axis('off')
plt.show()
