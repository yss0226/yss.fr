# -*- coding: utf-8 -*-
 
import numpy as np 
import cv2
 
fourcc = cv2.VideoWriter_fourcc("D", "I", "B", " ")
out = cv2.VideoWriter('frame_mosic.MP4',fourcc, 20.0, (640,480))
 
cv2.namedWindow("CaptureFace")
#调用摄像头
cap=cv2.VideoCapture(0)
#人眼识别器分类器
classfier=cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt.xml")
while cap.isOpened():
    read,frame=cap.read()
    if not read:
        break
    #灰度转换
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #人脸检测
    Rects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(Rects) > 0:            
        for Rect in Rects:  
             x, y, w, h = Rect  
             # 打码：使用高斯噪声替换识别出来的人眼所对应的像素值
             frame[y+10:y+h-10,x:x+w,0]=np.random.normal(size=(h-20,w))
             frame[y+10:y+h-10,x:x+w,1]=np.random.normal(size=(h-20,w))
             frame[y+10:y+h-10,x:x+w,2]=np.random.normal(size=(h-20,w))
 
    cv2.imshow("CaptureFace",frame)
    if cv2.waitKey(5)&0xFF==ord('q'):
        break
    # 保存视频
    out.write(frame)
#释放相关资源
cap.release()
out.release()
cv2.destroyAllWindows()
