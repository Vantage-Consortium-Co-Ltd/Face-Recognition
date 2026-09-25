import cv2
from Knn import knn, Create_Data, FACE_BOX

def main():
    X, y = Create_Data()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    x1, y1, x2, y2 = FACE_BOX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = frame[y1:y2, x1:x2, :]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 26, 125), 2)


        label = knn(X, y, gray.flatten(), k=4)

        cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 26, 125), 2)

        cv2.imshow("face", gray)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
