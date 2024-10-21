# Import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import Sort # ignore
from utilFOR import get_car, read_license_plate, write_csv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import csv
import datetime

# ------------------------- CSV FILE HANDLING FUNCTIONS --------------------------

# Function to write the final results into a CSV file
def final_csvfile(csv_path):
    file = open(csv_path, "r")
    file_reader = csv.reader(file)
    data_list = list(file_reader)
    final_list = data_list[1:]  # Skip the header row
    final_dict = {}

    for data in final_list:
        sub_dict1 = {
            'license_number_score': data[6],
            'license_plate_bbox': data[3],
            'license_number': data[5],
            'license_plate_bbox_score': data[4]
        }
        sub_dict2 = {'car_bbox': data[2]}
        final_dict[data[1]] = {'license_plate': sub_dict1, 'car': sub_dict2}

    with open("output.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['CarID', 'License Number Score', 'License Plate Bbox', 'License Number', 'License Plate Bbox Score', 'Car Bbox'])
        for car_id, data in final_dict.items():
            license_plate = data['license_plate']
            car_bbox = data['car']['car_bbox']
            writer.writerow([
                car_id,
                license_plate['license_number_score'],
                license_plate['license_plate_bbox'],
                license_plate['license_number'],
                license_plate['license_plate_bbox_score'],
                car_bbox
            ])

# Function to read the allowed vehicles from a CSV file

def is_allowed(license_plate, csv_file='allowed_cars.csv'):
    with open(csv_file, "r") as file:
        file_reader = csv.reader(file)
        next(file_reader)  # Skip the header row
        for row in file_reader:
            if row[0] == license_plate:
                return True  # License plate found in the CSV file
    return False  # License plate not found in the CSV file


# ------------------------- EMAIL NOTIFICATION FUNCTION --------------------------

# Function to send email with summary of unauthorized vehicles
def send_summary_email(unauthorized_cars, receiver='admin@parkinglot.com'):
    sender_email = 'jatayu.kavach@gmail.com'
    sender_password = 'quwkviupmsbgfzfk'
    receiver_email = receiver

    # Compose the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'End-of-Day Unauthorized Vehicle Report'

    # Create the body with unauthorized vehicle details
    custom_message = "Unauthorized vehicles detected today:\n\n"
    for plate, time in unauthorized_cars:
        custom_message += f"License Plate: {plate} | Time Detected: {time}\n"

    msg.attach(MIMEText(custom_message, "plain"))

    # Send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

# ------------------------- MAIN LICENSE PLATE DETECTION FUNCTION --------------------------

def plate_recog():
    results = {}
    unauthorized_cars = []  # Store unauthorized vehicles and detection time
    mot_tracker = Sort()

    # Load YOLO models
    coco_model = YOLO('Models\yolov8n.pt')
    license_plate_detector = YOLO('Models\license_plate_detector.pt')

    # Load video feed
    cap = cv2.VideoCapture('Traffic.mp4')

    vehicles = [2, 3, 5, 7]  # Vehicle classes
    frame_nmr = -1
    ret = True

    # Process each frame
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if ret:
            results[frame_nmr] = {}

            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                detections_.append([x1, y1, x2, y2, score])

            # Track detected vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                bbox = (int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1))

                # Associate detected license plate with a vehicle
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    cvzone.cornerRect(frame, bbox, l=9)

                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)

                    cv2.imwrite('license_plate_crop.jpg', license_plate_crop)
                    cv2.imwrite('license_plate_thresh.jpg', license_plate_crop_thresh)

                    # OCR to read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        # Check if the vehicle is allowed
                        if not is_allowed(license_plate_text, "Data\\allowed.csv"):
                            # Store unauthorized vehicles and detection time
                            unauthorized_cars.append((license_plate_text, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text,
                                              'bbox_score': score, 'text_score': license_plate_text_score}
                        }

        if ret == False:
            break

    # Save final results
    final_csvfile('D:\\University\\All Projects\\CCTV_analysis_system\\Data\\test.csv')
    write_csv(results, 'D:\\University\\All Projects\\CCTV_analysis_system\\Results\\output.csv')

    # Send summary email with unauthorized vehicles at the end of the day
    if unauthorized_cars:
        send_summary_email(unauthorized_cars)

# ------------------------- MAIN FUNCTION --------------------------

def main():
    try:
        print("Starting the license plate recognition system...")
        plate_recog()
        print("License plate recognition completed. Results have been saved.")
    except Exception as e:
        print(f"An error occurred during license plate recognition: {e}")

if __name__ == "__main__":
    main()
