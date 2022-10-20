"""
命令行用法
		python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
		python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
"""

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def sound_alarm(path):
    # 读取音频文件 发出警报声
    playsound.playsound(path)


"""
eye_aspect_ratio(eye)：eye即给定的眼睛面部标志的x/y坐标
	1.A和B分别是计算两组垂直眼睛标志之间的距离，而C是计算水平眼睛标志之间的距离。
	  A：P2到P6的距离。B：P3到P5的距离。C：P1到P4的距离。
	  eye：眼睛面部标志的x/y坐标包含了P1到P6一共6个x/y坐标，索引从0开始，即eye[0]为P1。
	2.计算眼睛纵横比，然后将眼图长宽比返回给调用函数。
		ear = (A + B) / (2.0 * C)
		ear = ((P2-P6) + (P3-P5)) / (2.0 * (P1-P4))
		分子中计算的是眼睛的特征点在垂直方向上的距离，分母计算的是眼睛的特征点在水平方向上的距离。
		由于水平点只有一组，而垂直点有两组，所以分母乘上了2，以保证两组特征点的权重相同。
"""


def eye_aspect_ratio(eye):
    # 计算两组垂直的坐标之间的欧式距离：计算眼睛中垂直的A(P2到P6的距离) 和 B(P3到P5的距离)
    A = dist.euclidean(eye[1], eye[5])  # eye[1]到eye[5]的欧式距离 即 P2到P6的欧式距离
    B = dist.euclidean(eye[2], eye[4])  # eye[2]到eye[4]的欧式距离 即 P3到P5的欧式距离
    # 计算一组水平的坐标之间的欧式距离：计算眼睛中水平的C(P1到P4的距离)
    C = dist.euclidean(eye[0], eye[3])  # eye[0]到eye[3]的欧式距离 即 P1到P4的欧式距离
    # 计算眼睛长宽比：(A + B) / (2.0 * C) 即 ((P2-P6) + (P3-P5)) / (2.0 * (P1-P4))
    ear = (A + B) / (2.0 * C)
    # 返回眼睛长宽比
    return ear


"""
构造参数解析并解析参数
	1.第一个命令行参数(必须)：--shape-predictor 这是dlib的预训练面部标志检测器的路径。
	2.第二个是参数(可选)：--video 它控制驻留在磁盘上的输入视频文件的路径。如果您想要使用实时视频流，则需在执行脚本时省略此开关。
"""
ap = argparse.ArgumentParser()
# 面部坐标位置预测器模型文件：shape_predictor_68_face_landmarks.dat
ap.add_argument("-p", "--shape-predictor", required=False, default="./shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
# 警报音频WAV文件：alarm.wav
ap.add_argument("-a", "--alarm", type=str, required=False, default="./alarm.wav", help="path alarm .WAV file")
# 网络摄像头在系统上的索引：代表的是使用第几个的摄像头设备
ap.add_argument("-w", "--webcam", type=int, required=False, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

"""
定义两个常数：
	EYE_AR_THRESH常数：眼睛的长宽比的阈值表示眨眼的阈值，或者说判断是否闭上眼的眼睛长宽比的阈值
	EYE_AR_CONSEC_FRAMES常数：眼睛连续闭合的帧数触发警报的阈值，如果眼睛连续闭合的帧数大于触发警报的阈值的话则发出音频警告
"""
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# 初始化帧计数器
COUNTER = 0
# 布尔值，用于指示警报是否消失
ALARM_ON = False

"""
1.初始化dlib的面部检测器（基于HOG），然后创建面部界标预测器。
2.当确定视频流中是否发生眨眼时，我们需要计算眼睛的长宽比。
  如果眼睛长宽比低于一定的阈值，然后超过阈值，那么我们将记录一个“眨眼”，
  EYE_AR_THRESH是眼睛的长宽比的阈值表示眨眼的阈值，我们默认它的值为 0.3，您也可以为自己的应用程序调整它。
  另外，我们有一个重要的常量EYE_AR_CONSEC_FRAME是眼睛连续闭合的帧数触发警报的阈值，这个值被设置为 3。
  表明眼睛长宽比小于0.3时，接着三个连续的帧一定发生眼睛持续闭合动作。
3.同样，取决于视频的帧处理吞吐率，您可能需要提高或降低此数字以供您自己实施。
  接着初始化两个计数器，COUNTER帧计数器是眼睛长宽比小于EYE_AR_THRESH的连续帧的总帧数，
  还可以额外设置TOTAL用于记录脚本运行时发生的眨眼的总次数。
  现在我们输入、命令行参数和常量都已经写好了，接着可以初始化dlib的人脸检测器和面部标志检测器。
"""

print("[INFO] 加载面部界标预测器...")
detector = dlib.get_frontal_face_detector()  # 获取正面人脸检测器
predictor = dlib.shape_predictor(args["shape_predictor"])  # 人脸形状预测器：shape_predictor_68_face_landmarks.dat

# 分别获取左眼和右眼的面部标志的索引，为下面的左眼和右眼提取（x，y）坐标的起始和结束数组切片索引值
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # 面部中的左眼的起始坐标和结束坐标
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # 面部中的右眼的起始坐标和结束坐标

# 启动视频流线程，决定是否使用基于文件的视频流或实时USB/网络摄像头/ Raspberry Pi摄像头视频流
print("[INFO] 启动视频流线程...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

"""
树莓派相机模块，取消注释：# vs = VideoStream(usePiCamera=True).start()。
如果您未注释上述两个，你可以取消注释# fileStream = False 以表明你是不是从磁盘读取视频文件。
在while处我们开始从视频流循环帧，循环播放视频流中的帧。
如果正在访问视频文件流，并且视频中没有剩余的帧，则从循环中断。
从我们的视频流中读取下一帧，然后调整大小并将其转换为灰度。然后，我们通过dlib内置的人脸检测器检测灰度帧中的人脸。
我们现在需要遍历帧中的每个人脸，然后对其中的每个人脸应用面部标志检测：
"""
while True:
    # 从线程视频文件流中抓取帧，调整其大小，然后将其转换为灰度通道
    frame = vs.read()
    # print(frame.shape) #(480, 640, 3)
    # resize设置width=450时，可以无需同时设置height，因为height会自动根据width所设置的值按照原图的宽高比例进行自适应地缩放调整到合适的值
    frame = imutils.resize(frame, width=450)
    # print(frame.shape) #(337, 450, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为单通道的灰度图
    # print(gray.shape) #(337, 450) 表示单通道的灰度图

    # 根据正面人脸检测器 在灰度框中检测人脸
    rects = detector(gray, 0)  # 默认值0 即可，表示采样次数
    # print(rects) # 比如 rectangles[[(179, 198) (266, 285)]]

    # 循环每个检测出来的人脸
    for rect in rects:
        # print(rect) #比如 [(208, 198) (294, 285)]

        # 确定面部区域的面部界标，然后将面部界标（x，y）坐标转换为NumPy数组
        # 传入gray灰度图、还有从灰度图中检测出来的人脸坐标rect
        shape = predictor(gray, rect)
        # print(shape) #比如 <dlib.full_object_detection object at 0x000001909A7C5BF0>
        # 将从灰度图中检测出来的人脸坐标转换为NumPy数组
        shape = face_utils.shape_to_np(shape)
        # print(shape) #NumPy数组值

        # 从人脸坐标转换后的NumPy数组中 提取左眼和右眼坐标，然后使用该坐标计算两只眼睛的眼睛纵横比
        leftEye = shape[lStart:lEnd]  # 面部中的左眼的起始坐标和结束坐标
        rightEye = shape[rStart:rEnd]  # 面部中的右眼的起始坐标和结束坐标
        leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼的眼睛纵横比
        rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼的眼睛纵横比

        # 将两只眼睛的眼睛纵横比一起求平均
        ear = (leftEAR + rightEAR) / 2.0

        """
        shape确定面部区域的面部标志，接着将这些（x，y）坐标转换成NumPy阵列。
        使用数组切片技术，我们可以分别为左眼left eye和右眼提取（x，y）坐标，然后我们计算每只眼睛的眼睛长宽比。
        下一个代码块简单地处理可视化眼部区域的面部标志。
        """
        # 计算左眼和右眼的凸包，然后可视化每只眼睛
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        """
        检查眼睛宽高比是否低于眨眼阈值，如果是，则增加眨眼帧计数器中的值。
        我们已经计算了我们的（平均的）眼睛长宽比，但是我们并没有真正确定是否发生了眨眼，这在下一部分中将得到关注。
        第一步检查眼睛纵横比是否低于我们的眨眼阈值，如果是，我们递增指示正在发生眨眼的连续帧数。
        否则，我们将处理眼高宽比不低于眨眼阈值的情况，我们对其进行检查，看看是否有足够数量的连续帧包含低于我们预先定义的阈值的眨眼率。
         如果检查通过，我们增加总的闪烁次数。然后我们重新设置连续闪烁次数 COUNTER。
        """

        # EYE_AR_THRESH是眼睛的长宽比的阈值表示眨眼的阈值，我们默认它的值为 0.3，如果眼睛的长宽比小于了0.3则表示眨眼
        if ear < EYE_AR_THRESH:
            # 连续眨眼总次数+= 1
            COUNTER += 1
            print(COUNTER)
            # EYE_AR_CONSEC_FRAMES常数：眼睛连续闭合的帧数触发警报的阈值，如果眼睛连续闭合的帧数大于触发警报的阈值的话则发出音频警告
            # 如果眼睛闭上足够的连续帧数，则发出警报
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # 如果警报未打开，则将其打开
                if not ALARM_ON:
                    ALARM_ON = True  # 开启警报标志

                    # 检查是否提供了警报文件，如果有，请启动一个线程以在后台播放警报声音
                    if args["alarm"] != "":
                        # 开启增加线程
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        """
                        在脚本运行过程中有一个主线程，若在主线程中创建了子线程，
                        当主线程结束时根据子线程daemon属性值的不同可能会发生下面的两种情况之一：
                            1.如果某个子线程的daemon属性为False，主线程结束时会检测该子线程是否结束，如果该子线程还在运行，
                              则主线程会等待它完成后再退出；
                            2.如果某个子线程的daemon属性为True，主线程运行结束时不对这个子线程进行检查而直接退出，
                              同时所有daemon值为True的子线程将随主线程一起结束，而不论是否运行完成。
                            3.属性daemon的值默认为False，如果需要修改，必须在调用start()方法启动线程之前进行设置。
                              另外要注意的是，上面的描述并不适用于IDLE环境中的交互模式或脚本运行模式，
                              因为在该环境中的主线程只有在退出Python IDLE时才终止。
                            4.守护线程不能持有任何需要关闭的资源，例如打开文件等，因为当主线程关闭时，子线程也会自动同时关闭，
                              守护线程没有任何机会来关闭文件，这会导致数据丢失。
                        """
                        t.deamon = True  # 开启守护进程deamon：作用为主线程关闭时，子线程也会自动同时关闭，至此程序整个停止了
                        # 线程启动
                        t.start()

                # 在图像上发出警报的文字
                cv2.putText(frame, "睡意警告", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 否则，眼睛的纵横比不低于眨眼阈值，因此请重置计数器并发出警报
        else:
            # 连续眨眼总次数 清0
            COUNTER = 0
            # 取消警报
            ALARM_ON = False

        # 在帧上绘制计算出的眼睛长宽比，以帮助调试和设置正确的眼睛长宽比阈值和帧计数器
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示图像
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 如果按下“ q”键，则退出循环
    if key == ord("q"):
        break

# 做清理
cv2.destroyAllWindows()
vs.stop()
