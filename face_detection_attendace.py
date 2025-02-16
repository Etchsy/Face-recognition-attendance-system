import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition

path = "E:\\micro proj\\ATTENDANCE\\attendace\\image_folder"
url = 'http://192.168.38.158/cam-hi.jpg'

# Use the correct path for checking Attendance.csv
attendance_folder_path = "E:\\micro proj\\ATTENDANCE"

# Attendance file path
attendance_file = os.path.join(attendance_folder_path, "Attendance.csv")

# Check if Attendance.csv exists or create it
if os.path.exists(attendance_file):
    print("Attendance file exists.")
else:
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)
    print("Attendance file created.")

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Class names loaded:", classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList


def markAttendance(name):
    with open(attendance_file, 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            f.flush()


encodeListKnown = findEncodings(images)
print('Encoding Complete')

while True:
    try:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)  # Adjusted tolerance
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            # Debug output for face distance and match status
            print(f"Face distances: {faceDis}")
            print(f"Best match index: {matchIndex}, Distance: {faceDis[matchIndex]}, Match: {matches[matchIndex]}")

            # Slightly increase distance threshold to allow for slight mismatches
            if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                name = classNames[matchIndex].upper()
                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "UNAUTHORIZED"
                color = (0, 0, 255)  # Red for unauthorized faces

            # Draw the rectangle and display the name on the frame
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Log attendance for both recognized and unauthorized faces
            markAttendance(name)

        # Show the image in the window
        cv2.imshow('Webcam', img)
        key = cv2.waitKey(5)
        if key == ord('q'):  # Press 'q' to quit
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cv2.destroyAllWindows()
