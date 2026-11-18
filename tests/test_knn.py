"""เทสต์ KNN, การอ่านรูป และการแบ่ง train/test (ไม่ต้องใช้กล้อง ไม่ใช้รูปหน้าจริง)
รูปในเทสต์เป็นภาพสีเรียบที่สร้างขึ้นใน tmp_path แล้วลบทิ้งเอง
รัน: pytest"""
import numpy as np

from evaluate import split_per_person
from knn import imwrite, knn, load_faces

SIZE = (18, 14)     # (สูง, กว้าง) ภาพเล็ก ๆ พอให้เทสต์เร็ว


# ---------- knn ----------
def test_knn_เลือกเพื่อนบ้านที่ใกล้ที่สุด():
    X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]], dtype=np.float32)
    y = np.array(['a', 'a', 'b', 'b'])
    assert knn(X, y, np.array([0.5, 0.2]), k=1) == 'a'
    assert knn(X, y, np.array([10.5, 9.8]), k=1) == 'b'


def test_knn_เสียงข้างมากชนะ_แม้ตัวที่ใกล้สุดเป็นอีกคน():
    X = np.array([[0.0], [1.5], [2.0], [2.5]], dtype=np.float32)
    y = np.array(['a', 'b', 'b', 'b'])
    # z = 0.4 ใกล้ 'a' ที่สุด แต่ k=3 จะได้ a, b, b -> 'b'
    assert knn(X, y, np.array([0.4]), k=1) == 'a'
    assert knn(X, y, np.array([0.4]), k=3) == 'b'


# ---------- load_faces ----------
def save_gray(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(path), np.full(SIZE, value, dtype=np.uint8))


def test_load_faces_อ่านตามโฟลเดอร์และข้ามไฟล์ที่ไม่ใช่รูป(tmp_path):
    save_gray(tmp_path / 'alice' / '0.jpg', 30)
    save_gray(tmp_path / 'alice' / '1.jpg', 40)
    save_gray(tmp_path / 'bob' / '0.jpg', 200)
    (tmp_path / 'bob' / 'note.txt').write_text('ไม่ใช่รูป', encoding='utf-8')
    save_gray(tmp_path / '.hidden' / '0.jpg', 0)       # โฟลเดอร์ซ่อน ต้องข้าม
    (tmp_path / 'readme.txt').write_text('ไฟล์นอกโฟลเดอร์คน ต้องข้าม', encoding='utf-8')

    X, y = load_faces(str(tmp_path))
    assert y.tolist() == ['alice', 'alice', 'bob']
    assert X.shape == (3, SIZE[0] * SIZE[1])
    assert X.dtype == np.float32


def test_load_faces_ระยะไม่ล้นเพราะเป็น_float(tmp_path):
    # ถ้ายังเป็น uint8 แล้วลบกัน 0 - 255 จะวนกลับเป็น 1 ทำให้รูปดำกับรูปขาวดู "ใกล้กัน"
    save_gray(tmp_path / 'dark' / '0.jpg', 0)
    save_gray(tmp_path / 'light' / '0.jpg', 255)
    X, y = load_faces(str(tmp_path))
    assert knn(X, y, X[1] - 1, k=1) == 'light'
    assert np.sum((X[0] - X[1]) ** 2) == 255 ** 2 * X.shape[1]


# ---------- split_per_person ----------
def test_split_ทุกคนมีรูปทั้ง_train_และ_test():
    y = np.array(['a'] * 10 + ['b'] * 3 + ['c'] * 1)
    tr, te = split_per_person(y, 0.2)
    assert not np.any(tr & te)                          # ไม่มีรูปไหนอยู่ทั้งสองฝั่ง
    assert np.all(tr | te)
    assert te[y == 'a'].sum() == 2                      # 20% ของ 10
    assert te[y == 'b'].sum() == 1                      # อย่างน้อย 1 รูปเสมอ
    assert tr[y == 'a'].any() and tr[y == 'b'].any()


def test_split_สุ่มซ้ำได้ผลเดิมเมื่อ_seed_เท่ากัน():
    y = np.array(['a'] * 20 + ['b'] * 20)
    assert np.array_equal(split_per_person(y, 0.2, seed=1)[1], split_per_person(y, 0.2, seed=1)[1])


def test_knn_บนรูปจริงที่สร้างขึ้น_จำได้ถูกทุกคน(tmp_path):
    rng = np.random.default_rng(0)
    for name, base in [('alice', 40), ('bob', 130), ('carol', 220)]:
        for i in range(5):
            img = np.clip(base + rng.normal(0, 8, SIZE), 0, 255).astype(np.uint8)
            (tmp_path / name).mkdir(exist_ok=True)
            imwrite(str(tmp_path / name / f'{i}.jpg'), img)
    X, y = load_faces(str(tmp_path))
    tr, te = split_per_person(y, 0.2)
    pred = np.array([knn(X[tr], y[tr], z, k=3) for z in X[te]])
    assert np.array_equal(pred, y[te])


def test_load_faces_ชื่อคนเป็นภาษาไทยก็อ่านได้(tmp_path):
    # cv2.imread / cv2.imwrite ใช้ path ภาษาไทยบน Windows ไม่ได้ -> load_faces ต้องใช้ตัวช่วยแทน
    save_gray(tmp_path / 'เมฆ' / '1.jpg', 90)
    save_gray(tmp_path / 'มิว' / '1.jpg', 180)
    X, y = load_faces(str(tmp_path))
    assert sorted(y.tolist()) == ['มิว', 'เมฆ']
    assert X.shape[0] == 2
