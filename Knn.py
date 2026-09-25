import numpy as np
import cv2

import os

FACE_BOX = (200, 60, 440, 360)  # (x1, y1, x2, y2) capture box shared by rec_train.py and realtime.py
FACE_SIZE = (FACE_BOX[2] - FACE_BOX[0], FACE_BOX[3] - FACE_BOX[1])  # (width, height)


def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]


def Create_Data(size=FACE_SIZE):
    y = []
    X = []
    for f in os.listdir():
        if not os.path.isfile(f) and not f.startswith("."):
            for i in os.listdir(f):
                if i.endswith(".jpg"):
                    x = cv2.imread(f + "/" + i, cv2.IMREAD_GRAYSCALE)
                    x = cv2.resize(x, size)
                    X.append(x.flatten())
                    y.append(f)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y)
    return X, y


