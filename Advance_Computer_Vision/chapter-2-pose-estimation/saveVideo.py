import cv2

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 600, 560)
# 捕获相机
cap = cv2.VideoCapture(0)
# *'mp4v' 解包操作 == 'm','p','v','4'
# 创键文件格式对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 创键VideoWriter
vw = cv2.VideoWriter('PoseVideos/output.mp4', fourcc, 30, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 读到了数据,把这一帧的图片写入到缓存中，释放的时候写入指定文件
    vw.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
vw.release()
cv2.destroyAllWindows()