import matplotlib.pyplot as plt
import face_recognition
import cv2

frame = cv2.imread("image/obama.jpg")

# 缩小图像以加快速度
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# 找到人脸
face_locations = face_recognition.face_locations(small_frame,model="cnn")

for top, right, bottom, left in face_locations:
    # 提取边界框在原图比例的边界框
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

# 提取人脸
face_image = frame[top:bottom, left:right]

# 高斯模糊人脸
face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

# 原图人脸替换
frame[top:bottom, left:right] = face_image

# 展示图像
img = frame[:, :, ::-1]
plt.axis('off')
plt.imshow(img)
