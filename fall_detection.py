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


def send_email_fall(image_path, time_string):
    sender_email = 'jatayu.kavach@gmail.com'
    sender_password = 'quwkviupmsbgfzfk'
    receiver_email = 'denildubariya18@gmail.com'

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Fall Detected at {time_string}'
    custom_message=f'The fall is detected at {time_string}'
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)
    msg.attach(MIMEText(custom_message, "plain"))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

def fall():
    cap = cv2.VideoCapture(0)
    # cap.set(3, 1280)
    # cap.set(4, 720)
    model = YOLO("YOLO-PretrainedModels//yolov8n.pt")
    classnames = ["person"]

    fall_start_time = None
    person_ID = []
    listSize = temp = 0
    mailed = False

    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    fall_detected = False
    while True:
        fall_detected = False
        success, img = cap.read()
        results = model(img, stream=True)

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

                if cls == 0 and conf > 0.3 and w > h:
                    fall_detected = True
                    cvzone.cornerRect(img, bbox, l=9)
                    cvzone.putTextRect(img, f'Fall Detected {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # resultTracker = tracker.update(detections)
        # for res in resultTracker:
        #     x1, y1, x2, y2, id = res
        #     if person_ID.count(id) == 0:
        #         person_ID.append(id)
        if fall_detected and fall_start_time is None:
            fall_start_time = time.time()
            mailed=False
            # listSize = len(person_ID)

        if not fall_detected and fall_start_time is not None:
            fall_start_time = None

        if fall_detected and fall_start_time is not None and (time.time()-fall_start_time) > 10 and mailed is False:
            """and temp != listSize"""
            cv2.imwrite('fallImage.jpg', img)
            current_time = datetime.datetime.now().time()
            time_string = current_time.strftime('%H:%M:%S')
            send_email_fall('fallImage.jpg',time_string)
            temp = listSize
            fall_start_time=None
            mailed = True

        cv2.imshow("Image", img)
        cv2.waitKey(1)