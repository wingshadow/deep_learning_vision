import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=5, minDetectionCon=0.5, minTrackingCon=0.5, thickness=1, radius=2):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon
        self.thickness = thickness
        self.radius = radius

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # 进一步细化眼睛和嘴唇坐标False
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, False,
                                                 self.minDetectionCon, self.minTrackingCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=self.thickness, circle_radius=self.radius)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpecs,
                                               self.drawSpecs)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                    # print(id, x, y, z)
                    face.append([x, y, z])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
