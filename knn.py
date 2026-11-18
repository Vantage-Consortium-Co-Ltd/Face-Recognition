import os

import cv2
import numpy as np


def imread_gray(path):
    """อ่านรูปขาวดำ (ใช้แทน cv2.imread เพราะ cv2.imread อ่าน path ภาษาไทยบน Windows ไม่ได้ จะได้ None)"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def imwrite(path, img):
    """บันทึกรูป (ใช้แทน cv2.imwrite ด้วยเหตุผลเดียวกัน: ชื่อคนเป็นภาษาไทยก็บันทึกได้)"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok


def knn(X, y, z, k=3):
    d = np.sum((X - z) ** 2, axis=1)            # ระยะห่าง (square euclidean) กับทุกภาพ
    idx = np.argsort(d)[:k]                     # k ภาพที่ใกล้ที่สุด
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]                 # ชื่อที่ได้โหวตมากสุด


def load_faces(data_dir):
    """อ่านทุก .jpg ใน data_dir/<ชื่อ>/ -> X (1 แถว = 1 ภาพที่ flatten แล้ว), y (ชื่อโฟลเดอร์)"""
    X, y = [], []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder) or name.startswith(('.', '__')):
            continue
        for f in sorted(os.listdir(folder)):
            if f.endswith('.jpg'):
                img = imread_gray(os.path.join(folder, f))
                if img is None:
                    continue
                X.append(img.flatten())
                y.append(name)
    # float32: ถ้าเป็น uint8 ตอนลบกันค่าจะติดลบแล้ววนกลับ (overflow) ระยะจะผิด
    return np.array(X, dtype=np.float32), np.array(y)
