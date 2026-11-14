"""วัดความแม่นของ KNN จากรูปที่เก็บไว้ (ไม่ต้องใช้กล้อง): python evaluate.py [โฟลเดอร์ข้อมูล]
แบ่งรูปของแต่ละคน 80% เป็นข้อมูลอ้างอิง 20% เป็นข้อสอบ แล้วลองหลายค่า k"""
import os
import sys

import numpy as np

from config import DATA_DIR
from knn import knn, load_faces

TEST_RATIO = 0.2
KS = [1, 3, 5, 7, 9]


def split_per_person(y, ratio, seed=0):
    """แบ่ง test ทีละคน จะได้ทุกคนมีรูปทั้งใน train และ test"""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), dtype=bool)
    for name in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == name))
        test[idx[:max(1, int(len(idx) * ratio))]] = True
    return ~test, test


os.chdir(os.path.dirname(os.path.abspath(__file__)))
X, y = load_faces(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
if len(np.unique(y)) < 2:
    raise SystemExit('ต้องมีรูปอย่างน้อย 2 คน')
names, counts = np.unique(y, return_counts=True)
print('ข้อมูล:', dict(zip(names.tolist(), counts.tolist())))

tr, te = split_per_person(y, TEST_RATIO)
print(f'train={tr.sum()} test={te.sum()}\n')
print(f'{"k":>3} {"accuracy":>9}   ' + '  '.join(f'{n}' for n in names))
for k in KS:
    pred = np.array([knn(X[tr], y[tr], z, k) for z in X[te]])
    per_person = [np.mean(pred[y[te] == n] == n) for n in names]   # recall ของแต่ละคน
    print(f'{k:>3} {np.mean(pred == y[te]):>9.3f}   ' + '  '.join(f'{p:.2f}' for p in per_person))
