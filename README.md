# Face-Recognition

Real-time face recognition from a webcam using a K-Nearest Neighbors (KNN) classifier written with only NumPy and OpenCV. Each face is a grayscale image flattened into a vector of raw pixels.

## Files

| File | Purpose |
|------|---------|
| `Knn.py` | `knn(X, y, z, k)` classifier and `Create_Data()` dataset loader |
| `rec_get_data.py` | Captures face images from the webcam and defines the shared `FACE_BOX` crop region |
| `rec_test.py` | Loads the dataset and recognizes faces live from the webcam |

## How it works

1. **Collect data** – `rec_get_data.py` crops the region `FACE_BOX = (x1, y1, x2, y2)` (240x300 px), converts it to grayscale, and saves a `.jpg` each time you press `s`.
2. **Load data** – `Create_Data()` reads every `.jpg` in each sub-folder of the working directory. The folder name becomes the label. It returns `X` (one flattened image per row, e.g. `(N, 72000)`) and `y` (a NumPy array of labels).
3. **Predict** – `knn()` computes the squared Euclidean distance from the live face to every stored image, takes the `k` nearest, and returns the majority label.

KNN has no real training step: the "model" is the stored dataset, and all the work happens at prediction time.

## Usage

```bash
pip install numpy opencv-python
```

**1. Capture images** – set `name` in `rec_get_data.py` to the label of the person, then run:

```bash
python rec_get_data.py
```

Align the face inside the rectangle and press `s` to save an image. Repeat for each person, changing `name` each time. Images are saved to `./<name>/face<i>.jpg`.

> Re-running restarts the counter at 1 and overwrites existing `face<i>.jpg` files. Use a new `name` or move old images first.

**2. Recognize** – run from the folder that contains the label sub-folders:

```bash
python rec_test.py
```

The predicted label is drawn above the box. Press `q` to quit.

## Choosing K

`rec_test.py` calls `knn(..., k=...)`.

- **Small K (1-3)** – sensitive to blurry or mislabeled images; the output may flicker between names.
- **Large K** – smoother but can underfit. If K exceeds the number of images of a person, that person can never win the vote.
- Use an **odd K** to avoid ties, and keep it below the smallest number of images per person.
- Speed barely changes with K, because distances to all images are always computed.

## Limitations

- Raw pixels are sensitive to lighting, pose, and face position.
- There is no "unknown person" handling: every face is assigned to one of the known labels.
- No train/test split or accuracy measurement yet.

## Privacy

Face images are personal data and are **not** committed to this repository. Dataset folders and `.jpg` files are listed in `.gitignore`. Keep your own images locally and do not push them.
