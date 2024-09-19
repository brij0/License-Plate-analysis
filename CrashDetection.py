from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import datetime
# from  sort import *
import numpy as np

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText


def send_email_crash(image_path,str):
    sender_email = 'jatayu.kavach@gmail.com'
    sender_password = 'quwkviupmsbgfzfk'
    receiver_email = 'denildubariya18@gmail.com'

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Crash Detected at time:{str}'

    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

def crash():
    cap = cv2.VideoCapture('./Crash-Detection/crash2.mp4')
    # cap.set(3, 1280)
    # cap.set(4, 720)
    model = YOLO("YOLO-PretrainedModels//yolov8l.pt")
    classnames = ['person','bicycle','car','motorcycle','airplane','bus','train','truck']
    classList=[1,2,3,4,5,6,7,8]
    crashDetected=False
    listSize = temp = 0
    mailed = False
    crash_start_time =None
    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    while True:
        success, img = cap.read()
        if success==False:
            break
        results = model(img, stream=True)
        crashDetected=False
        detections = np.empty((0, 5))

        for r in results:

            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                # print(x1, y1, x2, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2-x1, y2-y1
                w, h = int(w), int(h)
                bbox = x1, y1, w, h

                conf = math.ceil((box.conf[0]*100))/100

                cls = box.cls[0]

                if cls in classList and conf > 0.5:
                    cvzone.cornerRect(img, bbox, l=9)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
                    for i in range(len(detections)-1):
                        p1=detections[i][:]
                        p2=detections[i+1][:]
                        print(p1,p2,crashDetected)
                        if ((((p1[0] < p2[0] < p1[2]) or (p1[0] < p2[2] < p1[2])) or ((p2[0] < p1[0] < p2[2]) or (p2[0] < p1[2] < p2[2]))) and (((p1[1] < p2[1] < p1[3]) or (p1[1] < p2[3] < p1[3])) or ((p2[1] < p1[1] < p2[3]) or (p2[1] < p1[3] < p2[3])))):
                            crashDetected=True
                if crashDetected:    
                    cvzone.putTextRect(img, f' Crash Detected  {conf} {crashDetected}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
        if crashDetected and crash_start_time is None:
            crash_start_time = time.time()
            # listSize = len(person_ID)

        if not crashDetected and crash_start_time is not None:
            crash_start_time = None

        if crashDetected and crash_start_time is not None and (time.time()-crash_start_time) > 3 and mailed is False:
            """and temp != listSize"""
            cv2.imwrite('crashimg.jpg', img)
            current_time = datetime.datetime.now().time()
            time_string = current_time.strftime('%H:%M:%S')
            send_email_crash('crashimg.jpg',time_string)
            temp = listSize
            mailed = True
        cv2.imshow("Image", img)
        cv2.waitKey(0)
