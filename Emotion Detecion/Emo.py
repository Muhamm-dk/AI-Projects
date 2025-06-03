import cv2 as cv
import mediapipe as mp
from deepface import DeepFace

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv.VideoCapture(0)

while True:
    key, img = cap.read()

    #rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)


    emotion = results[0]['dominant_emotion']
    cv.putText(img, f'Emotion: {emotion}', (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.imshow('Emotion Recognition', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv.destroyAllWindows()

