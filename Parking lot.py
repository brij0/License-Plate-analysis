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
def send_email_license(image_path, license_plate, color, model, time_of_notice, receiver='admin@parkinglot.com'):
    sender_email = 'youremail@gmail.com'
    sender_password = 'yourpassword'
    receiver_email = receiver

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Unauthorized Vehicle Detected: {license_plate}'
    
    # Custom message with vehicle details
    custom_message = f'Unauthorized vehicle detected. License Plate: {license_plate}\nColor: {color}\nModel: {model}\nTime: {time_of_notice}'
    
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)
    msg.attach(MIMEText(custom_message, "plain"))

    # Send email through SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, r eceiver_email, msg.as_string())
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
    coco_model = YOLO('./YOLO-PretrainedModels/yolov8n.pt')  # Pretrained YOLO model for car detection
    license_plate_detector = YOLO('license_plate_detector.pt')  # Custom YOLO model for license plate detection

    # Load video for real-time processing (change path for live feed if needed)
    cap = cv2.VideoCapture('./videos/ParkingLot.mp4')

    # Vehicles class IDs in COCO dataset (cars, motorcycles, buses, etc.)
    vehicles = [2, 3, 5, 7]

    allowed_cars = read_allowed_cars()  # Read the list of allowed cars from CSV
    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            # Detect vehicles in the frame
            detections = coco_model(frame)[0]
            detections_ = []

            # Draw detection boxes around detected vehicles
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                w, h = int(x2) - int(x1), int(y2) - int(y1)
                bbox = int(x1), int(y1), int(w), int(h)

                if int(class_id) in vehicles:  # If detection is a vehicle
                    detections_.append([x1, y1, x2, y2, score])
                    cvzone.cornerRect(frame, bbox, l=9)
                    
                    # License plate detection for each vehicle
                    license_plates = license_plate_detector(frame)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate

                        # Crop license plate area for recognition
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)

                        # Extract license plate number using OCR
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                        if license_plate_text:  # If a license plate is recognized
                            print(f"Detected License Plate: {license_plate_text}")
                            
                            # Check if vehicle is allowed
                            if not is_allowed(license_plate_text, allowed_cars):
                                # If not allowed, save info
                                car_color = "Unknown"  # Color and model detection can be added if required
                                car_model = "Unknown"
                                time_of_notice = time.strftime("%Y-%m-%d %H:%M:%S")
                                
                                unauthorized_cars[license_plate_text] = {
                                    "color": car_color,
                                    "model": car_model,
                                    "time": time_of_notice
                                }

                                # Save image of the unauthorized vehicle
                                cv2.imwrite(f'{license_plate_text}_unauthorized.jpg', frame)
                                
                                # Send email to notify about unauthorized vehicle
                                send_email_license(f'{license_plate_text}_unauthorized.jpg', license_plate_text, car_color, car_model, time_of_notice)

            # Display the current frame
            cv2.imshow("Parking Lot", frame)
            cv2.waitKey(1)

    # Close the video and windows
    cap.release()
    cv2.destroyAllWindows()


# Call the main function
plate_recog()
