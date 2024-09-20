from geopy.geocoders import Nominatim
from ultralytics import YOLO
import cv2
import cvzone
import utilIND
from sort.sort import *
from utilIND import get_car, read_license_plate, write_csv

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import csv

def final_csvfile(csv_path):
    file = open(csv_path, "r")
    file_reader = csv.reader(file)
    data_list = list(file_reader)
    final_list = data_list[1:]
    final_dict = {}
    for data in final_list:
        sub_dict1 = {}
        sub_dict2 = {}
        sub_dict3 = {}
        sub_dict1['license_number_score'] = data[6]
        sub_dict1['license_plate_bbox'] = data[3]
        sub_dict1['license_number'] = data[5]
        sub_dict1['license_plate_bbox_score'] = data[4]
        sub_dict2['car_bbox'] = data[2]
        sub_dict3['license_plate'] = sub_dict1
        sub_dict3['car'] = sub_dict2
        final_dict[data[1]] = sub_dict3
    csv_file_path = "output.csv"
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header
        writer.writerow(['CarID', 'License Number Score', 'License Plate Bbox',
                        'License Number', 'License Plate Bbox Score', 'Car Bbox'])

        # Write the data
        for car_id, data in final_dict.items():
            license_plate = data['license_plate']
            license_plate_bbox = license_plate['license_plate_bbox']
            car_bbox = data['car']['car_bbox']

            writer.writerow([
                car_id,
                license_plate['license_number_score'],
                license_plate_bbox,
                license_plate['license_number'],
                license_plate['license_plate_bbox_score'],
                car_bbox
            ])


def read_data():
    
    file = open('data.csv', "r")
    file_reader = csv.reader(file)
    data_list = list(file_reader)
    final_list = data_list[1:]
    final_dict = {}
    for row in final_list:
        final_dict[row[0]]=row[1]
    return final_dict



def send_email_license(image_path, speed,receiver='denildubariya18@gmail.com'):
    sender_email = 'jatayu.kavach@gmail.com'
    sender_password = 'quwkviupmsbgfzfk'
    receiver_email = receiver

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Overspeed Challan {speed}'
    custom_message = f'You were caught overspeeding at the speed of {speed} now you have to pay the chalan'
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)
    msg.attach(MIMEText(custom_message, "plain"))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

def plate_recog():
    results = {}

    mot_tracker = Sort()

    # load models
    coco_model = YOLO('./YOLO-PretrainedModels/yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')

    # load video
    cap = cv2.VideoCapture('./videos/Traffic.mp4')
    # cap = cv2.VideoCapture(2)

    vehicles = [2, 3, 5, 7]
    limit1=[0,137,1300,137]
    limit2=[0,557,1300,557]
    distance = 180
    # read frames
    frame_nmr = -1
    ret = True
    brea=True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            cv2.line(frame,(limit1[0],limit1[1]),(limit1[2],limit1[3]),(0,0,255),5)
            cv2.line(frame,(limit2[0],limit2[1]),(limit2[2],limit2[3]),(0,255,0),5)
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                w, h = int(x2)-int(x1), int(y2)-int(y1)
                bbox = int(x1), int(y1), int(w), int(h)
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score]) 
                    cvzone.cornerRect(frame, bbox, l=9)
                    cx,cy=int(x1)+w//2,int(y1)+h//2
                    cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
                    if(limit1[0]<cx<limit1[2] and limit1[1]-10<cy<limit1[1]+10):
                        detection_start_time = time.time()
                    elif (limit2[0] < cx < limit2[2] and limit2[1]-10 < cy < limit2[1]+10):
                        total_time=time.time()-detection_start_time
                        actual_time=total_time/2.8333333
                        speed=distance/actual_time
                        if(speed>40):
                            # coordinates=get_coordinates()

                            cv2.imwrite('overspeed.jpg', frame)
                            send_email_license('overspeed.jpg',speed)
                            print("\n\n\n\n",speed,"\n\n\n\n\n")
            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                # w, h = int(x2)-int(x1), int(y2)-int(y1)
                # bbox = int(x1), int(y1), int(w), int(h)

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    cvzone.cornerRect(frame, bbox, l=9)

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)
                    cv2.imshow('original_crop',license_plate_crop)
                    cv2.imshow('threshold', license_plate_crop_thresh)
                    cv2.waitKey(1)
                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    print("\n\n\n\nCar\n\n\n\n\n",license_plate_text,"\n\n\n\n\n\n")
                    brea=False

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
        if ret==False:
            break
        elif ret==True:
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
    write_csv(results,'test.csv')
    final_csvfile('test.csv')