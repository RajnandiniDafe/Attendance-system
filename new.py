import math
import time
import cv2
import cvzone
from ultralytics import YOLO
from AttendanceProject import markAttend

confidence = 0.6

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("models/l_version_1_250.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])

            if conf > confidence:
                if classNames[cls] == 'real':
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                # Draw the bounding box and display the class name
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

                # Only mark attendance for real faces, ignore fake/spoofed ones
                try:
                    if classNames[cls] == 'real':
                        crop_img = img[y1:y1 + h, x1:x1 + w]
                        print('REAL FACE DETECTED')
                        cv2.imwrite('/Users/rajnandinidafe/Desktop/final year project/Attendance-system/image1.png', crop_img)
                        markAttend(crop_img)  # Pass the correct image path
                    else:
                        print('SPOOF DETECTED')
                except Exception as e:
                    print(f"Error: {e}")

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.imshow("Image", img)
    cv2.waitKey(1)
