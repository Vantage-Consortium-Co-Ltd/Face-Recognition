import numpy as np
import cv2
import os

def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]


def Create_Data():
    y = []
    X = []
    for f in os.listdir():
        if not os.path.isfile(f) and not f.startswith("."):
            for i in os.listdir(f):
                if i.endswith(".jpg"):
                    x = cv2.imread(f + "/" + i, cv2.IMREAD_GRAYSCALE)
                    X.append(x.flatten())
                    y.append(f)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    return X, y

