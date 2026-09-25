import cv2
import numpy as np
import os

def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

y = []
X = []
for f in os.listdir():
    if not os.path.isfile(f) and not f.startswith("."):
        for i in os.listdir(f):
            if i.endswith(".jpg"):
                x = cv2.imread(f"{f}/{i}", cv2.IMREAD_GRAYSCALE)
                X.append(x.flatten())
                y.append(f)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(sorted(set(y)))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 1600, 900)
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (465,135), (815,585), (0,0,255), 2)
    face = cv2.cvtColor(frame[135:585, 465:815, :], cv2.COLOR_BGR2GRAY)
    name = knn(X, y, face.flatten())
    cv2.putText(frame, name, (465,125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
