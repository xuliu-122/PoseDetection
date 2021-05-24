import cv2
import numpy as np


def empty(a):
    pass


def StackImages(scale, imgArray): #功能：实现图片的连接
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]),None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


#chapter 1
'''
# 1、图像的读写
print("Package Imported")
img = cv2.imread("Resourse/1.jpg")
cv2.imshow("Output", img)
cv2.waitKey(0)
# 2、视频的读取
cap = cv2.VideoCapture("Resourse/1.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("Videooutput", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 3、调用摄像头，获取视频
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
while True:
    success, img = cap.read()
    cv2.imshow("Videooutput", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
# chapter2
'''
img = cv2.imread("Resourse/2.jpg")#读取图像
kernel = np.ones((5, 5), np.uint8)#生成一个5x5的内核矩阵

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换RGB
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)#加入高斯噪声
imgCanny = cv2.Canny(img, 150, 200)#Canny边缘检测
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)#膨胀
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)#侵蚀

cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Erode Image", imgEroded)
cv2.waitKey(0)
'''
# chapter3
'''
img = cv2.imread("Resourse/3.jpg")
print(img.shape)

imgResize = cv2.resize(img, (400, 500))#重新设置大小
print(imgResize.shape)

imgCropped = img[200:978, 0:1024]#裁剪图片 
print(imgCropped.shape)

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)
'''
# chapter4
'''
img = np.zeros((512, 512, 3), np.uint8)
print(img.shape)
print(img)
img[100:200, 100:200] = 255, 0, 0
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)#画线
cv2.rectangle(img, (0, 0), (200, 400), (0, 0, 255), 3)#画矩形
cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)#画圆
cv2.putText(img, "OpenCV I'm coming!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 150, 0), 3)#放置文本

cv2.imshow("Image", img)
cv2.waitKey(0)
'''
# chapter5
'''
img = cv2.imread("Resourse/4.jpg")
width, height = 300, 300
pts1 = np.float32([[475,80], [709, 384], [94, 270], [303, 625]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("Image", img)
cv2.imshow("Output", imgOutput)
cv2.waitKey(0)
'''
# chapter6
# 用函数实现图像拼接
'''
img = cv2.imread("Resourse/6.jpg")
imgStack = StackImages(0.5, ([img], [img], [img]))
cv2.imshow("ImageStack", imgStack)
cv2.waitKey(0)
'''
'''用numpy实现图像拼接，注意channel数量
img = cv2.imread("Resourse/5.jpg")
print(img.shape)

imgHor = np.hstack((img, img))
print(imgHor.shape)
imgVer = np.vstack((img, img))
print(imgVer.shape)
cv2.imshow("Image", img)
cv2.imshow("HorizontalImage", imgHor)  
cv2.imshow("VerticalImage", imgVer)
cv2.waitKey(0)
'''
# chapter7
'''
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 260)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 45, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 54, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread("Resourse/6.jpg")
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    imgstack = StackImages(0.6, ([img, imgHSV], [mask, imgResult]))

    # print(h_min)

    # cv2.imshow("Image", img)
    # cv2.imshow("HSVImage", imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("ImageResult", imgResult)
    cv2.imshow("ImageStack", imgstack)
    cv2.waitKey(25)
'''
# chapter8

'''
def getContours(img):
    countours, Hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 400:
            cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            #图形分类
            if objCor == 3: objectType = "Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:objectType = "Square"#利用纵横比来进行正方形的判断
                else:objectType = "Rectangle"
            elif objCor >4: objectType = "Circles"
            else:objectType = "None"
            #将分类的图像标签显示在检测框中
            cv2.rectangle(imgContours, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imgContours, objectType, (x+(w//2)-10, y+(h//2)-10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1)

img = cv2.imread("Resourse/15.jpg")
imgContours = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)

imgBlank = np.zeros_like(img)

imgStack = StackImages(0.8, ([img, imgGray, imgBlur], [imgCanny, imgContours, imgBlank]))

# cv2.imshow("Image", img)
# cv2.imshow("GrayImage", imgGray)
# cv2.imshow("BlurImage", imgBlur)
cv2.imshow("ImageStack", imgStack)
cv2.waitKey(0)

'''
# chapter9
'''
faceCascade = cv2.CascadeClassifier("C:/Users/52810/PycharmProjects/pose/Resourse/haarcascade_frontalface_default.xml")
img = cv2.imread("Resourse/10.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
'''

