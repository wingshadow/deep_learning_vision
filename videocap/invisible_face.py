import cv2

cap = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, img = cap.read()
    faces = face_detect.detectMultiScale(img, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("camera", img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
