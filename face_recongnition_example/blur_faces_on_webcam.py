import face_recognition
import cv2

# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    # 将视频帧大小调整为 1/4 大小以加快人脸检测处理
    # InputArray src ：输入，原图像，即待改变大小的图像；
    # OutputArray dst： 输出，改变后的图像。这个图像和原图像具有相同的内容，只是大小和原图像不一样而已；
    # dsize：输出图像的大小，如上面例子（300，300）。
    #
    # 其中，fx和fy就是下面要说的两个参数，是图像width方向和height方向的缩放比例。
    # fx：width方向的缩放比例
    # fy：height方向的缩放比例
    # 将视频帧的大小调整为1/4以加快面部检测处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    """
        face_locations(img, number_of_times_to_upsample=1, model='hog') 
            给定一个图像，返回图像中每个人脸的面部特征位置(眼睛、鼻子等) 
            参数： 
                img：一个image（numpy array类型） 
                number_of_times_to_upsample：从images的样本中查找多少次人脸，该参数值越高的话越能发现更小的人脸。 
                model：使用哪种人脸检测模型。
                    “hog” 准确率不高，但是在CPUs上运行更快，“cnn” 更准确更深度（且 GPU/CUDA加速，如果有GPU支持的话），默认是“hog” 
                返回值： 一个元组列表，列表中的每个元组包含人脸的位置(top, right, bottom, left)
        """
    # Find all the faces and face encodings in the current frame of video
    # 查找当前视频帧中的所有面部位置和面部位置编码
    face_locations = face_recognition.face_locations(small_frame, model="hog")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # 由于我们检测到的帧被缩放到 1/4 大小，因此放大了人脸位置
        # 由于我们在中检测到的帧被缩放到1/4大小，因此此处需要对检测出来的人脸位置(top, right, bottom, left)重新放大4倍
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Extract the region of the image that contains the face
        # 根据放大4倍后还原到原图规模的人脸位置(top, right, bottom, left)到视频原始帧中 进行提取包含人脸的图像区域
        face_image = frame[top:bottom, left:right]

        # Blur the face image
        # 使用高斯模糊来模糊面部图像
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image
        # 将模糊的人脸区域放回帧图像中
        frame[top:bottom, left:right] = face_image

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
