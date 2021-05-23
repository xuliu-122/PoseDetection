import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideos/3.mp4')  # 使用自己的视频数据
# cap = cv2.VideoCapture(0)  #调用电脑摄像头
pTime = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = detector.FindPose(img)
    lmList = detector.FindPosition(img)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (255, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)