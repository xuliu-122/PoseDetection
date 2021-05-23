# 可以实现人体关键骨骼点的检测（33），详细的33节点可以查询mediapipe的官方文档查看
# 接下来将会对以下代码进行力所能及的注释，希望可以帮助到大家
#首先导入所需要的函数库
import cv2
import mediapipe as mp
import time

#将mediapipe中的一些方法引出

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# cap = cv2.VideoCapture('PoseVideos/3.mp4')#使用自己的视频数据
cap = cv2.VideoCapture(0)  #调用电脑摄像头
pTime = 0 #初始化pTime
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#因为图像读取进来是BGR，而之后的代码处理是使用RGB，所以这里的将BGR转换为RGB
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)#获取处理完成后得到的人体骨骼点，并连线
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape #获取图片的一些数据，这里的
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)#转换为实际图像的长宽
            cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)#这里是用原点将图像中检测出来的骨骼点绘制出来

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)#将实时监测的帧率打印在图像的左上角（50,50）位置
    cv2.imshow("Image", img)
    cv2.waitKey(1)

