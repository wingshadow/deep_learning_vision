import face_recognition

# 加载图像
known_obama_image = face_recognition.load_image_file("image/obama.jpg")
known_biden_image = face_recognition.load_image_file("image/biden.jpg")

# 获得人脸图像特征
obama_face_encoding = face_recognition.face_encodings(known_obama_image)[0]
biden_face_encoding = face_recognition.face_encodings(known_biden_image)[0]

known_encodings = [
    obama_face_encoding,
    biden_face_encoding
]

# 加载未知人脸图像
image_to_test = face_recognition.load_image_file("image/obama2.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# 计算未知人脸和已知人脸的距离
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

# 查看不同距离阈值下的人脸匹配结果
for i, face_distance in enumerate(face_distances):
    # 打印距离
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    # 当阈值为0.6，是否匹配
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    # 当阈值为更严格的0.5，是否匹配
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(
        face_distance < 0.5))
    print()