"""เก็บรูปหน้าจากกล้อง: python collect.py <ชื่อ>
จัดหน้าให้อยู่ในกรอบแดง กด s = บันทึกภาพ, q = ออก"""
import os
import sys

import cv2

from config import DATA_DIR, TOP, BOTTOM, LEFT, RIGHT

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2:
    raise SystemExit('ใส่ชื่อคนด้วย เช่น python collect.py mek')
name = sys.argv[1]
folder = os.path.join(DATA_DIR, name)
os.makedirs(folder, exist_ok=True)

# นับต่อจากภาพที่มีอยู่แล้ว จะได้ไม่บันทึกทับ
i = len([f for f in os.listdir(folder) if f.endswith('.jpg')]) + 1

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        print('อ่านภาพจากกล้องไม่ได้')
        break

    face = cv2.cvtColor(frame[TOP:BOTTOM, LEFT:RIGHT], cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (0, 0, 255), 2)
    cv2.putText(frame, f'{name}: {i - 1} images', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):          # กด s = บันทึกภาพ
        cv2.imwrite(os.path.join(folder, f'{i}.jpg'), face)
        print('saved', i)
        i += 1
    elif key == ord('q'):        # กด q = ออก
        break

cap.release()
cv2.destroyAllWindows()
