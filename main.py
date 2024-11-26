import cv2
import cvzone
from ultralytics import YOLO
import numpy as np 


cap = cv2.VideoCapture('video.MOV')
model = YOLO('yolov8n-pose.pt')

while True:
    ret,frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('video.MOV')
        continue

    frame = cv2.resize(frame,(640,720))
    width,height = frame.shape[:2]
    blank_image = np.zeros((width,height,3),dtype=np.uint8)

    results = model(frame)
    frame = results[0].plot()

    for keypoints in results[0].keypoints.data:
        keypoints = keypoints.cpu().numpy()

        for i,keypoint in enumerate(keypoints):
            x,y,confidence = keypoint

            if confidence > 0.7:
                cv2.circle(blank_image,(int(x),int(y)),radius=5,color=(255,0,0),thickness=1)
                cv2.putText(blank_image,f'{i}',(int(x),int(y)-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,
                            (255,255,255),1,cv2.LINE_AA)
                
        connections = [
            (3, 1), (1, 0), (0, 2), (2, 4), (1, 2),(4, 6),(3,5),
            (5, 6), (5, 7), (7, 9),         
            (6, 8), (8, 10),                
            (11, 12), (11, 13), (13, 15),   
            (12, 14), (14, 16),             
            (5, 11), (6, 12)               
        ]
        
        for part_a,part_b in connections:
            x1,y1,conf1 = keypoints[part_a]
            x2,y2,conf2 = keypoints[part_b]

            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(blank_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,255),
                         thickness=2)

    output = cvzone.stackImages([frame,blank_image],cols=2,scale=0.80)
    cv2.imshow('frame',output)
    # cv2.imshow('frames',blank_image)
    if cv2.waitKey(0) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()