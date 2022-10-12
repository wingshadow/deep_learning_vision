import matplotlib.pylab as plt
import face_recognition
import cv2
from scipy.spatial import distance as dist

# 这是一个检测眼睛状态的演示
# 人眼闭上次数超过设定阈值EYES_CLOSED_SECONDS，判定人眼处于闭眼状态
EYES_CLOSED_SECONDS = 2


def main():
    # 闭眼次数
    closed_count = 0
    # 读取两张图像模仿人睁闭眼
    img_eye_opened = cv2.imread('image/eye_open.jpg')
    img_eye_closed = cv2.imread('image/eye_close.jpg')
    # 设置图像输入序列，前1张睁眼，中间3张闭眼，最后1张睁眼
    frame_inputs = [img_eye_opened] + [img_eye_closed] * 3 + [img_eye_opened] * 1

    for frame_num, frame in enumerate(frame_inputs):
        # 缩小图片
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # bgr通道变为rgb通道
        rgb_small_frame = small_frame[:, :, ::-1]
        # 人脸关键点检测
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        # 没有检测到关键点
        if len(face_landmarks_list) < 1:
            continue

        # 获得人眼特征点位置
        for face_landmark in face_landmarks_list:
            # 每只眼睛有六个关键点，以眼睛最左边顺时针顺序排列
            left_eye = face_landmark['left_eye']
            right_eye = face_landmark['right_eye']

            # 计算眼睛的纵横比ear，ear这里不是耳朵的意思
            ear_left = get_ear(left_eye)
            ear_right = get_ear(right_eye)
            # 判断眼睛是否闭上
            # 如果两只眼睛纵横比小于0.2，视为眼睛闭上
            closed = ear_left < 0.2 and ear_right < 0.2
            # 设置眼睛检测闭上次数
            if closed:
                closed_count += 1
            else:
                closed_count = 0
            # 如果眼睛检测闭上次数大于EYES_CLOSED_SECONDS，输出眼睛闭上
            if closed_count > EYES_CLOSED_SECONDS:
                eye_status = "frame {} | EYES CLOSED".format(frame_num)
            elif closed_count > 0:
                eye_status = "frame {} | MAYBE EYES CLOSED ".format(frame_num)
            else:
                eye_status = "frame {} | EYES OPENED ".format(frame_num)
            print(eye_status)

            plt.imshow(rgb_small_frame)
            # 左右眼轮廓第一个关键点颜色为red，最后一个关键点颜色为blue，其他关键点为yellow
            color = ['red'] + ['yellow'] * int(len(left_eye) - 2) + ['blue']
            # 按照顺序依次绘制眼睛关键点
            for index in range(len(left_eye)):
                leye = left_eye[index]
                reye = right_eye[index]
                plt.plot(leye[0], leye[1], 'bo', color=color[index])
                plt.plot(reye[0], reye[1], 'bo', color=color[index])
                plt.title(eye_status)

            plt.show()


# 计算人眼纵横比
def get_ear(eye):
    # 计算眼睛轮廓垂直方向上下关键点的距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算水平方向上的关键点的距离
    C = dist.euclidean(eye[0], eye[3])

    # 计算眼睛的纵横比
    ear = (A + B) / (2.0 * C)

    # 返回眼睛的纵横比
    return ear


if __name__ == "__main__":
    main()
