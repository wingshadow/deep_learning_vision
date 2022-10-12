import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import face_recognition

# 通过PIL加载图片
biden_image = face_recognition.load_image_file("image/biden.jpg")
obama_image = face_recognition.load_image_file("image/obama.jpg")
unknown_image = face_recognition.load_image_file("image/obama2.jpg")

plt.imshow(biden_image)
plt.title('biden')
plt.axis('off')
plt.show()
plt.imshow(obama_image)
plt.title('obama')
plt.axis('off')
plt.show()
plt.imshow(unknown_image)
plt.title('unknown')
plt.axis('off')
plt.show()

# 获取输入图像文件中每个人脸的人脸特征，人脸特征维度为128
# 由于输入图像中可能有多张脸，因此它会返回一个特征列表。
# 默认输入图像只有一张人脸，只关心每个图像中的第一个特征，所以设置特征获取索引为0
# 建议单步看看该函数运行机制
try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    # 没有找到人脸的情况
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

# 已知人脸列表，按照顺序为拜登的人脸特征，奥巴马的人脸特征
known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

# 如果未知人脸与已知人脸数组中的某个人匹配，则匹配结果为真
# 这个函数调用了face_distance人脸特征距离计算函数，可以单步调试看看源代码
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

# 是否和第一个人匹配
print("Is the unknown face a picture of Biden? {}".format(results[0]))
# 是否和第二个人匹配
print("Is the unknown face a picture of Obama? {}".format(results[1]))
# 这张人脸是否曾经见过
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
