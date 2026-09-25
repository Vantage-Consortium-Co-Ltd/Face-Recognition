import cv2
import os
from Knn import FACE_BOX

def capture_face():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    name = 'Mew'
    i = 1
    os.makedirs(name, exist_ok=True)

    x1, y1, x2, y2 = FACE_BOX

    while True:
        ret, frame = cap.read()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 26, 125), 2)
        face = frame[y1:y2, x1:x2, :]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        print(gray.flatten().shape)
        cv2.imshow('face', gray)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f'{name}/face{i}.jpg', gray)
            i += 1

if __name__ == "__main__":
    capture_face()




        
