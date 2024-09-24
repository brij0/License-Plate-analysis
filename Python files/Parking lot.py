from geopy.geocoders import Nominatim
from ultralytics import YOLO
import cv2
import cvzone
import utilFOR
import numpy as np
from sort.sort import Sort
from utilFOR import get_car, read_license_plate, write_csv

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import csv
import time

# Function to read the data of allowed vehicles from a CSV file
def read_allowed_cars():
    # Reads allowed cars from a file
    file = open('allowed_cars.csv', "r")
    file_reader = csv.reader(file)
    data_list = list(file_reader)
    final_list = data_list[1:]
    allowed_cars = {}
    for row in final_list:
        allowed_cars[row[0]] = row[1]  # Assuming row[0] is license plate, row[1] is other info (e.g., owner name or model)
    return allowed_cars


# Function to send email notification for unauthorized parking
def send_email_license(image_path, license_plate,time_of_notice, receiver='admin@parkinglot.com'):
    sender_email = 'youremail@gmail.com'
    sender_password = 'yourpassword'
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


# Function to check if a vehicle is allowed to park in the lot
def is_allowed(license_plate, allowed_cars_dict):
    return allowed_cars_dict.get(license_plate, None) is not None


# Function for license plate recognition at parking lot entrance
def plate_recog():
    # Dictionary to hold unauthorized cars
    unauthorized_cars = {}

    mot_tracker = Sort()

    # Load YOLO models for object and license plate detection
    license_plate_detector = YOLO('yolov8n.pt')  # Custom YOLO model for license plate detection

    # Load video for real-time processing (change path for live feed if needed)
    cap = cv2.VideoCapture('Traffic.mp4')

    # Vehicles class IDs in COCO dataset (cars, motorcycles, buses, etc.)
    vehicles = [2, 3, 5, 7]

    allowed_cars = read_allowed_cars()  # Read the list of allowed cars from CSV
    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}

            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))
            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                w, h = int(x2)-int(x1), int(y2)-int(y1)
                bbox = int(x1), int(y1), int(w), int(h)

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
    final_csvfile('test.csv')
    write_csv(results,'test.csv')


def main():
    # Display message for starting the process
    print("Starting license plate recognition at the parking lot...")

    # Call the plate recognition function
    plate_recog()

    # Display message when the process is complete
    print("License plate recognition completed.")

if __name__ == "__main__":
    main()