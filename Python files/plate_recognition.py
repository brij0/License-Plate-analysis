from ultralytics import YOLO
import cv2
import cvzone
import datetime
import numpy as np
from sort import Sort
from utilFOR import *
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import csv

# ------------------------- CSV FILE HANDLING FUNCTIONS --------------------------

# Function to write the final results into a CSV file
def final_csvfile(csv_path):
    # Read existing CSV data
    file = open(csv_path, "r")
    file_reader = csv.reader(file)
    data_list = list(file_reader)
    final_list = data_list[1:]  # Skip the header row
    final_dict = {}

    # Iterate over the CSV rows and store them in a dictionary
    for data in final_list:
        sub_dict1 = {
            'license_number_score': data[6],
            'license_plate_bbox': data[3],
            'license_number': data[5],
            'license_plate_bbox_score': data[4]
        }
        sub_dict2 = {'car_bbox': data[2]}
        final_dict[data[1]] = {'license_plate': sub_dict1, 'car': sub_dict2}

    # Write the final dictionary back into a new CSV file
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

# Function to read the allowed cars from a CSV file
def read_allowed_cars(csv_file="Data\\allowed.csv"):
    allowed_cars = {}
    # Open the allowed cars CSV and store license plates
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            allowed_cars[row[0]] = row[1]  # License plate as key, owner info as value
    return allowed_cars

# ------------------------- EMAIL NOTIFICATION FUNCTION --------------------------

# Function to send email notifications for unauthorized vehicles
def send_email_license(image_path, license_plate, time_of_notice, receiver='admin@parkinglot.com'):
    sender_email = 'jatayu.kavach@gmail.com'
    sender_password = 'quwkviupmsbgfzfk'
    receiver_email = receiver

    # Compose the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Unauthorized Vehicle Detected: {license_plate}'

    # Custom message body with vehicle details
    custom_message = f'Unauthorized vehicle detected. License Plate: {license_plate}\nTime: {time_of_notice}'

    # Attach image of the license plate and the message
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    msg.attach(img)
    msg.attach(MIMEText(custom_message, "plain"))

    # Send email via SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

# ------------------------- MAIN LICENSE PLATE DETECTION FUNCTION --------------------------

# Function to perform license plate recognition and validation
def plate_recog():
    results = {}
    not_allowed_cars = {}  # Store unauthorized vehicles and detection time

    # Load the allowed cars from the CSV file
    allowed_cars = read_allowed_cars()

    # Initialize vehicle tracker (Sort algorithm)
    mot_tracker = Sort()

    # Load YOLO models for vehicle detection and license plate detection
    coco_model = YOLO('./YOLO-PretrainedModels/yolov8n.pt')
    license_plate_detector = YOLO('Models\license_plate_detector.pt')

    # Load video feed (can be a live camera feed or pre-recorded video)
    cap = cv2.VideoCapture('Data\Traffic.mp4')

    vehicles = [2, 3, 5, 7]  # Vehicle classes in COCO dataset
    frame_nmr = -1
    ret = True

    # Process each frame of the video
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if ret:
            results[frame_nmr] = {}

            # Detect vehicles in the frame
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                detections_.append([x1, y1, x2, y2, score])

            # Track the detected vehicles using Sort
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates within the vehicles
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                bbox = (int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1))

                # Associate the detected license plate with a vehicle
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Draw a bounding box around the detected license plate
                    cvzone.cornerRect(frame, bbox, l=9)

                    # Extract the license plate region from the frame
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # Convert the license plate to grayscale and apply thresholding
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)

                    # Save the processed license plate images
                    cv2.imwrite('license_plate_crop.jpg', license_plate_crop)
                    cv2.imwrite('license_plate_thresh.jpg', license_plate_crop_thresh)

                    # Use OCR to read the license plate text
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        # Check if the vehicle is allowed to park
                        if not is_allowed(license_plate_text, allowed_cars):
                            # Log unauthorized vehicles and detection time
                            not_allowed_cars[license_plate_text] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Send an email alert for unauthorized vehicle
                            send_email_license('license_plate_crop.jpg', license_plate_text, not_allowed_cars[license_plate_text])

                        # Store the detected car and license plate details
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                              'text': license_plate_text,
                                              'bbox_score': score,
                                              'text_score': license_plate_text_score}
                        }

        # Stop if there are no more frames to process
        if ret == False:
            break

    # Save final results to a CSV file
    final_csvfile('D:\\University\\All Projects\\CCTV_analysis_system\\Data\\test.csv')
    write_csv(results, 'Results\output.csv')

# ------------------------- MAIN FUNCTION --------------------------

def main():
    try:
        print("Starting the license plate recognition system...")
        plate_recog()
        print("License plate recognition completed. Results have been saved.")
    except Exception as e:
        print(f"An error occurred during license plate recognition: {e}")

# Run the main function
if __name__ == "__main__":
    main()
