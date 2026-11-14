"""จดจำหน้าแบบ real-time จากกล้อง: python recognize.py
กด q = ออก"""
import os

import cv2
import numpy as np

from config import DATA_DIR, TOP, BOTTOM, LEFT, RIGHT, K
from knn import knn, load_faces

os.chdir(os.path.dirname(os.path.abspath(__file__)))
X, y = load_faces(DATA_DIR)
if len(X) == 0:
    raise SystemExit('ไม่พบภาพ ให้รัน collect.py ก่อน')
names, counts = np.unique(y, return_counts=True)
print('ข้อมูล:', dict(zip(names.tolist(), counts.tolist())))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        print('อ่านภาพจากกล้องไม่ได้')
        break

    face = cv2.cvtColor(frame[TOP:BOTTOM, LEFT:RIGHT], cv2.COLOR_BGR2GRAY)
    name = knn(X, y, face.flatten().astype(np.float32), K)

    cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (0, 0, 255), 2)
    cv2.putText(frame, str(name), (LEFT, TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
