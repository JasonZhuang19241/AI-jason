import cv2
import os
import shutil
import face_recognition
import sys
import time
import win32com.client as wincl
import win32api
from datetime import datetime
date=datetime.date(datetime.now())
print("###############################################################################")
print("attendence for: "+str(date))
nameof=str(date)+".txt"
file1 = open(nameof,"a+")
mytime = time.localtime()
if mytime.tm_hour < 12:
    daytime='morning'
elif mytime.tm_hour<18:
    daytime='afternoon'
else:
    daytime='evening'
def say(ax):
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(ax)
list=[]
video=cv2.VideoCapture(0)
number_face=0
for file in os.listdir('known_faces'):
    number_face+=1
real=False
KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.4
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
recognized=[]
MODEL ='hog'#hog
def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
print('Loading known faces...')
known_faces = []
known_names = []
a=1
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
        write="["+"="*a+">"+" "*(number_face-a)+"] "+str(a)+"/"+str(number_face)
        sys.stdout.write("\r" + write)
        sys.stdout.flush()
        a+=1


print('Processing unknown faces...')
while True:
    ret,image=video.read()
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            real=True
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = name_to_color(match)
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            if len(recognized)<5:
                recognized.append(match)
            else:
                if recognized[0]==recognized[1]==recognized[2]==recognized[3]==recognized[4] and match not in list:
                    list.append(recognized[0])
                    print(list)
                    print(str(len(list))+"/"+str(a-1))
                    speak="Good "+daytime+" "+recognized[0]
                    say(speak)
                    writed=str(recognized[1])+", "
                    file1.write(writed)
                    print(writed)
                    recognized=[]
                else:
                    recognized=[]
            
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
file1.close()
video.release()    
cv2.destroyAllWindows()
print(list)
a=input()


