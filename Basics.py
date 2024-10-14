import cv2
import numpy as np
import face_recognition

imgPriyansh = face_recognition.load_image_file('ImagesBasic/Priyansh.png')
imgPriyansh = cv2.cvtColor(imgPriyansh, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/PriyanshTest.png')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgPriyansh) [0]
encodePriyansh = face_recognition.face_encodings(imgPriyansh) [0]
cv2.rectangle(imgPriyansh,(faceLoc[3], faceLoc[0]) , (faceLoc[1], faceLoc[2]) , (0, 255 , 255), 2)

faceLocTest = face_recognition.face_locations(imgTest) [0]
encodeTest = face_recognition.face_encodings(imgTest) [0]
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]) , (faceLocTest[1], faceLocTest[2]) , (0, 0 , 255), 2)

results = face_recognition.compare_faces([encodePriyansh], encodeTest)
faceDis = face_recognition.face_distance([encodePriyansh], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

cv2.imshow('Priyansh', imgPriyansh)
cv2.imshow('Priyansh', imgTest)
cv2.waitKey(0)