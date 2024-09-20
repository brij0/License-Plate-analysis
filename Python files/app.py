
import cv2
from flask import Flask, render_template, Response
from plate_recognition import *
from fall_detection import *
from CrashDetection import *
app = Flask(__name__)

# Function to capture frames from the video

def generate_frames_crash():

    cap = cv2.VideoCapture('./Crash-Detection/crash2.mp4')
    # cap.set(3, 1280)
    # cap.set(4, 720)
    model = YOLO("YOLO-PretrainedModels//yolov8l.pt")
    classnames = ['person', 'bicycle', 'car',
                  'motorcycle', 'airplane', 'bus', 'train', 'truck']
    classList = [1, 2, 3, 4, 5, 6, 7, 8]
    crashDetected = False
    listSize = temp = 0
    mailed = False
    crash_start_time = None
    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    while cap.isOpened():
        success, img = cap.read()
        if success == False:
            break
        results = model(img, stream=True)
        crashDetected = False
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
                        p1 = detections[i][:]
                        p2 = detections[i+1][:]
                        print(p1, p2, crashDetected)
                        if ((((p1[0] < p2[0] < p1[2]) or (p1[0] < p2[2] < p1[2])) or ((p2[0] < p1[0] < p2[2]) or (p2[0] < p1[2] < p2[2]))) and (((p1[1] < p2[1] < p1[3]) or (p1[1] < p2[3] < p1[3])) or ((p2[1] < p1[1] < p2[3]) or (p2[1] < p1[3] < p2[3])))):
                            crashDetected = True
                if crashDetected:
                    cvzone.putTextRect(img, f' Crash Detected  {conf} {crashDetected}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
        if crashDetected and crash_start_time is None:
            crash_start_time = time.time()
            mailed=False
            # listSize = len(person_ID)

        if not crashDetected and crash_start_time is not None:
            crash_start_time = None

        if crashDetected and crash_start_time is not None and (time.time()-crash_start_time) > 1 and mailed is False:
            cv2.imwrite('crashimg.jpg', img)
            current_time = datetime.datetime.now().time()
            time_string = current_time.strftime('%H:%M:%S')
            send_email_crash('crashimg.jpg', time_string)
            temp = listSize
            crash_start_time=None
            mailed = True
        success, buffer = cv2.imencode('.jpg', img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        img = buffer.tobytes()
        yield (b'--img\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
    cap.release()


def generate_frames_fall():
    # cap.set(4, 720)
    model = YOLO("YOLO-PretrainedModels//yolov8n.pt")
    classnames = ["person"]
    cap=cv2.VideoCapture(2)

    fall_start_time = None
    person_ID = []
    listSize  = 0
    fall_detected = False
    mailed = False
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
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
        if fall_detected and fall_start_time is None:
            fall_start_time = time.time()
            mailed=False

        if not fall_detected and fall_start_time is not None:
            fall_start_time = None

        if fall_detected and fall_start_time is not None and (time.time()-fall_start_time) > 10 and mailed is False:
            cv2.imwrite('fallImage.jpg', img)
            current_time = datetime.datetime.now().time()
            time_string = current_time.strftime('%H:%M:%S')
            send_email_fall('fallImage.jpg', time_string)
            temp = listSize
            fall_start_time=None
            mailed = True
        cv2.imshow("Image",img)
        cv2.waitKey(1)
        success, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--img\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
    cap.release()

def generate_frames_plate():
    # cap = cv2.VideoCapture('./videos/Traffic.mp4')
    global speed
    results = {}

    mot_tracker = Sort()

    # load models
    coco_model = YOLO('./YOLO-PretrainedModels/yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')

    # load video
    cap = cv2.VideoCapture('./videos/Traffic.mp4')
    # cap = cv2.VideoCapture(2)

    vehicles = [2, 3, 5, 7]
    limit1 = [0, 137, 1300, 137]
    limit2 = [0, 557, 1300, 557]
    distance = 180
    # read frames
    frame_nmr = -1
    ret = True
    brea = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1
        # ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            cv2.line(frame, (limit1[0], limit1[1]),
                     (limit1[2], limit1[3]), (0, 0, 255), 5)
            cv2.line(frame, (limit2[0], limit2[1]),
                     (limit2[2], limit2[3]), (0, 255, 0), 5)
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                w, h = int(x2)-int(x1), int(y2)-int(y1)
                bbox = int(x1), int(y1), int(w), int(h)
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    cvzone.cornerRect(frame, bbox, l=9)
                    cx, cy = int(x1)+w//2, int(y1)+h//2
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    if (limit1[0] < cx < limit1[2] and limit1[1]-10 < cy < limit1[1]+10):
                        detection_start_time = time.time()
                    elif (limit2[0] < cx < limit2[2] and limit2[1]-10 < cy < limit2[1]+10):
                        total_time = time.time()-detection_start_time
                        actual_time = total_time/3.8
                        speed = distance/actual_time
                        if (speed > 40):
                            # coordinates=get_coordinates()

                            cv2.imwrite('overspeed.jpg', frame)
                            send_email_license('overspeed.jpg', speed)
                            print("\n\n\n\n", speed, "\n\n\n\n\n")
            # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                # w, h = int(x2)-int(x1), int(y2)-int(y1)
                # bbox = int(x1), int(y1), int(w), int(h)

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                    license_plate, track_ids)

                if car_id != -1:
                    cvzone.cornerRect(frame, bbox, l=9)

                    # crop license plate
                    license_plate_crop = frame[int(
                        y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(
                        license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)
                    cv2.imshow('original_crop', license_plate_crop)
                    cv2.imshow('threshold', license_plate_crop_thresh)
                    cv2.waitKey(1)
                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(
                        license_plate_crop_thresh)
                    print("\n\n\n\nCar\n\n\n\n\n",
                          license_plate_text, "\n\n\n\n\n\n")
                    brea = False

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

        cv2.imshow("Image", frame)
        cv2.waitKey(1)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    write_csv(results, 'test.csv')
    final_csvfile('test.csv')
    cap.release()

@app.route('/white.html')
def home():
    return render_template('white.html')

@app.route('/')
def index():
    return render_template('white.html')

@app.route('/main_page.html')
def options():
    return render_template('main_page.html')

@app.route('/Number_plate.html')
def number_plate():
    return render_template('index.html')

@app.route('/video_feed_plate')
def video_feed_plate():
    return Response(generate_frames_plate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/forget_password.html')
def forgot():
    return render_template('forget_password.html')


@app.route('/signup.html')
def sign_up():
    return render_template('signup.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/fall_detection.html')
def fall_detection():
    return render_template('index_fall.html')

@app.route('/video_feed_fall')
def video_feed_fall():
    return Response(generate_frames_fall(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/crash.html')
def crash():
    return render_template('index_crash.html')


@app.route('/video_feed_crash')
def video_feed_crash():
    return Response(generate_frames_crash(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
