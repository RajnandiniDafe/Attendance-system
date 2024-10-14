import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = '/Users/rajnandinidafe/Desktop/final year project/Attendance-system/ImagesAttendance'
images = []
className = []

# List all files in the directory
myList = os.listdir(path)
print("Files found:", myList)

# Load images and their corresponding class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Warning: Image {cl} not loaded properly.")
        continue
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

print("Class Names:", className)

def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Ensure there's a face to encode
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face found in image.")
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        #does not repeat the names
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d-%m-%Y')
            f.writelines(f'\n{name},{dateString},{dtString}')
'''

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
'''

# Get encodings for all loaded images
encodeListKnown = findEncodings(images)
print("Number of encodings found:", len(encodeListKnown))


#cap = cv2.VideoCapture(0)

def markAttend(img):
    #cap = cv2.VideoCapture(0)
    #success, img = cap.read()
    # img = captureScreen()
    #print(f"SIZE: {len(img)}")
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)




