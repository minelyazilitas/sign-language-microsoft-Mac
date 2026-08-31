# MediaPipe'ın kaçırdığı kısa el boşluklarını komşu karelerden dolduruyoruz
import os
import sys
import json
import shutil
import datetime
import argparse
import numpy as np

MAX_GAP = 3
HANDS = {
    "sol": (703, [(66, 108), (150, 192)]),
    "sag": (704, [(108, 150), (192, 234)]),
}


def fill_one(X):
    X = X.copy()
    filled = 0

    for i in range(len(X)):
        for mask_col, ranges in HANDS.values():
            mask = X[i, :, mask_col]
            present = np.where(mask == 1)[0]
            if len(present) < 2:
                continue

            #  ardışık iki dolu kare arasında boşluk kontrolü
            for a, b in zip(present[:-1], present[1:]):
                gap = b - a - 1
                if gap == 0 or gap > MAX_GAP:
                    continue
                for t in range(a + 1, b):
                    w = (t - a) / (b - a)   # a'ya mi b'ye mi daha yakin
                    for lo, hi in ranges:
                        X[i, t, lo:hi] = X[i, a, lo:hi] * (1 - w) + X[i, b, lo:hi] * w
                    X[i, t, mask_col] = 1   # artık elimizde degeri var
                    filled += 1

    # konumlar değişti, hız ve ivmeyi yeniden hesaplıyoruz
    pos = X[:, :, 0:234]
    vel = np.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    acc = np.zeros_like(pos)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

    bos = X[:, :, 702:705].sum(axis=2) == 0
    vel[bos] = 0
    acc[bos] = 0

    X[:, :, 234:468] = vel
    X[:, :, 468:702] = acc

    return X, filled


ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description="kisa el bosluklarini doldurur")
    p.add_argument("--kaynak", default="islenmis_ham", help="okunacak veri klasoru")
    p.add_argument("--hedef", default="islenmis_dolu", help="yazilacak yeni klasor")
    a = p.parse_args()

    src = os.path.join(ROOT, "data", a.kaynak)
    dst = os.path.join(ROOT, "data", a.hedef)
    if not os.path.isdir(src):
        sys.exit(f"kaynak klasor yok: {src}")

    if os.path.exists(dst):
        sys.exit(f"bu klasor zaten var: {dst}\nbaska bir ad ver: --hedef <ad>")
    os.makedirs(dst)

    stats = {}
    for name in ["train", "val", "test"]:
        X = np.load(os.path.join(src, f"X_{name}.npy"))

        before = ((X[:, :, 703] + X[:, :, 704]) > 0).mean() * 100
        X, filled = fill_one(X)
        after = ((X[:, :, 703] + X[:, :, 704]) > 0).mean() * 100

        np.save(os.path.join(dst, f"X_{name}.npy"), X)

        shutil.copy(os.path.join(src, f"y_{name}.npy"),
                    os.path.join(dst, f"y_{name}.npy"))

        stats[name] = {"dolduruldu": int(filled),
                       "el_tespiti_once": round(before, 1),
                       "el_tespiti_sonra": round(after, 1)}
        print(f"{name}: {filled} kare dolduruldu, el tespiti %{before:.1f} -> %{after:.1f}")

    meta = {"adim": "bosluk doldurma", "kaynak": a.kaynak,
            "tarih": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "max_gap": MAX_GAP, "olcumler": stats}
    with open(os.path.join(dst, "kunye.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"bitti -> data/{a.hedef}")


if __name__ == "__main__":
    main()
