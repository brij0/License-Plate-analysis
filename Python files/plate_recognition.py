from geopy.geocoders import Nominatim
from ultralytics import YOLO
import cv2
import cvzone
import datetime
import numpy as np
from sort import Sort
from sort import *
from utilFOR import *

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

def read_allowed_cars(csv_file="D:\\University\\All Projects\\CCTV_analysis_system\\output\\allowed.csv"):
    allowed_cars = {}
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            allowed_cars[row[0]] = row[1]  # Assuming row[0] is license plate, row[1] is owner name or other info
    return allowed_cars

# Function to check if a vehicle is allowed to park in the lot
def is_allowed(license_plate, allowed_cars_dict):
    return allowed_cars_dict.get(license_plate, None) is not None

# Modified email function to send alert for not allowed cars
def send_email_license(image_path, license_plate, time_of_notice, receiver='admin@parkinglot.com'):
    sender_email = 'jatayu.kavach@gmail.com'
    sender_password = 'quwkviupmsbgfzfk'
    receiver_email = receiver

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Unauthorized Vehicle Detected: {license_plate}'

    # Custom message with vehicle details
    custom_message = f'Unauthorized vehicle detected. License Plate: {license_plate}\nTime: {time_of_notice}'
    
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)
    msg.attach(MIMEText(custom_message, "plain"))

    # Send email through SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()


def plate_recog():
    results = {}
    not_allowed_cars = {}  # Dictionary to store not allowed cars and their time of detection

    # Read the list of allowed cars from CSV
    allowed_cars = read_allowed_cars()

    mot_tracker = Sort()

    # load models
    coco_model = YOLO('./YOLO-PretrainedModels/yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')

    # load video
    cap = cv2.VideoCapture('Traffic.mp4')

    vehicles = [2, 3, 5, 7]
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                w, h = int(x2)-int(x1), int(y2)-int(y1)
                bbox = int(x1), int(y1), int(w), int(h)

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    cvzone.cornerRect(frame, bbox, l=9)

                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)
                    
                    # Save images instead of showing them
                    cv2.imwrite('license_plate_crop.jpg', license_plate_crop)
                    cv2.imwrite('license_plate_thresh.jpg', license_plate_crop_thresh)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        # Check if car is allowed
                        if not is_allowed(license_plate_text, allowed_cars):
                            # If car is not allowed, add to dictionary
                            not_allowed_cars[license_plate_text] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            send_email_license('license_plate_crop.jpg', license_plate_text, not_allowed_cars[license_plate_text])

                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

        if ret == False:
            break
    final_csvfile('D:\\University\\All Projects\\CCTV_analysis_system\\output\\test.csv')
    write_csv(results, 'D:\\University\\All Projects\\CCTV_analysis_system\\output\\test.csv')



def main():
    # Call the plate recognition function
    try:
        print("Starting the license plate recognition system...")
        plate_recog()
        print("License plate recognition completed. Results have been saved.")
    except Exception as e:
        print(f"An error occurred during license plate recognition: {e}")
        
if __name__ == "__main__":
    main()
