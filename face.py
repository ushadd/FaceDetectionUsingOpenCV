import cv2
import numpy as np
from deepface import DeepFace
import face_recognition

# Load pre-trained age and gender models

GENDER_PROTO = "C:\\Users\\Usha sri\\Downloads\\gender_deploy.prototxt"
GENDER_MODEL = "C:\\Users\\Usha sri\\Downloads\\gender_net.caffemodel"

gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
GENDER_LIST = ['Male', 'Female']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    for face in face_locations:
        top, right, bottom, left = face
        face_img = frame[top:bottom, left:right]
        
        # Emotion Detection
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        
        # Age & Gender Detection
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)
        gender = GENDER_LIST[np.argmax(gender_net.forward())]
        
        
        label = f"{gender},{emotion}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imshow('Face Recognition with Emotion, Gender', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
