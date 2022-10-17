import cv2
import mediapipe as mp
import time

cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 1920, 1080)

cap = cv2.VideoCapture("Videos/1.mp4")
pTime = 0
# google mediapipe 人脸提取模型
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.5)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            # 检测到的人脸的集合，其中每个人脸都表示为一个检测原始消息，其中包含 人脸的概率、
            # 1 个边界框、6 个关键点（右眼、左眼、鼻尖、嘴巴中心、右耳、左耳）。
            # 边界框由 xmin 和 width (由图像宽度归一化为 [0, 1])以及 ymin 和
            # height (由图像高度归一化为 [0, 1])组成。每个关键点由 x 和 y 组成，
            # 分别通过图像宽度和高度归一化为 [0, 1]

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            # 将边界框的坐标点从比例坐标转换成像素坐标
            # 将边界框的宽和高从比例长度转换为像素长度
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)
    cv2.imshow("video", img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()