"""
SignFlow / LSA64 — Landmark Çıkarma Modülü
LSA64 videolarından MediaPipe (Tasks API) ile pose + el landmark'larını çıkarır
ve model eğitimi için (N, 30, 225) boyutunda .npy matrislerine dönüştürür.

Çıktı formatı, eski parquet pipeline'ı ile birebir aynıdır:
    30 frame × 225 özellik   (75 landmark × 3 eksen)
    75 landmark = pose(33) + left_hand(21) + right_hand(21)

Kullanım:
    python src/extract_landmarks.py            # tüm 3200 video
    python src/extract_landmarks.py --limit 50 # hızlı test (ilk 50 video)
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2

# Proje kök dizinini ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Sabitler ---
FIXED_FRAMES = 30            # Her örnek sabit 30 frame
FEATURES_PER_FRAME = 225     # 75 landmark × 3 eksen (x, y, z)
N_POSE = 33
N_HAND = 21                  # tek el
# 225 = pose(33*3=99) + left_hand(21*3=63) + right_hand(21*3=63)

VIDEO_DIR = os.path.join(project_root, 'data', 'raw', 'lsa64', 'all_cut')
OUTPUT_DIR = os.path.join(project_root, 'data', 'processed')
MODELS_DIR = os.path.join(project_root, 'models')

POSE_MODEL = os.path.join(MODELS_DIR, 'pose_landmarker_lite.task')
HAND_MODEL = os.path.join(MODELS_DIR, 'hand_landmarker.task')

# Kişi-bağımsız split: kişi 1-8 train, 9 val, 10 test
TRAIN_SIGNERS = set(range(1, 9))
VAL_SIGNERS = {9}
TEST_SIGNERS = {10}


def build_landmarkers():
    """Pose ve Hand landmarker'larını oluşturur (IMAGE modunda)."""
    for path in (POSE_MODEL, HAND_MODEL):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model dosyası yok: {path}\n"
                "Gerekli .task model dosyalarını models/ klasörüne indirin."
            )

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)
    hand_lm = vision.HandLandmarker.create_from_options(hand_opts)
    return pose_lm, hand_lm


def _frame_to_features(rgb_frame, pose_lm, hand_lm):
    """Tek bir RGB frame'den 225 özellikli vektör üretir. Eksik landmark → 0."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    pose_vec = np.zeros(N_POSE * 3, dtype=np.float32)
    left_vec = np.zeros(N_HAND * 3, dtype=np.float32)
    right_vec = np.zeros(N_HAND * 3, dtype=np.float32)

    # --- Pose ---
    pose_res = pose_lm.detect(mp_image)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        pose_vec = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                            dtype=np.float32).flatten()

    # --- Eller (handedness ile sol/sağ ayrımı) ---
    hand_res = hand_lm.detect(mp_image)
    if hand_res.hand_landmarks:
        for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            vec = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                           dtype=np.float32).flatten()
            label = handed[0].category_name  # 'Left' veya 'Right'
            if label == 'Left':
                left_vec = vec
            else:
                right_vec = vec

    return np.concatenate([pose_vec, left_vec, right_vec])


def extract_from_video(video_path, pose_lm, hand_lm):
    """
    Bir videodan (FIXED_FRAMES, FEATURES_PER_FRAME) boyutunda landmark matrisi çıkarır.
    Video boyunca 30 frame eşit aralıkla örneklenir (cut videolar baştan sona harekettir).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return np.zeros((FIXED_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

    # Örneklenecek frame indeksleri
    if total >= FIXED_FRAMES:
        indices = np.linspace(0, total - 1, FIXED_FRAMES).astype(int)
    else:
        indices = np.arange(total)  # kısa video → sonra sıfır padding

    sequence = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            sequence.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sequence.append(_frame_to_features(rgb, pose_lm, hand_lm))

    cap.release()

    # 30'dan kısaysa sıfır ile doldur
    while len(sequence) < FIXED_FRAMES:
        sequence.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))

    return np.array(sequence[:FIXED_FRAMES], dtype=np.float32)


def parse_filename(path):
    """'001_003_002.mp4' → (sign=1, signer=3, rep=2). label = sign - 1."""
    name = os.path.splitext(os.path.basename(path))[0]
    sign, signer, rep = (int(p) for p in name.split('_'))
    return sign, signer, rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='Sadece ilk N videoyu işle (0 = hepsi). Hızlı test için.')
    args = parser.parse_args()

    print("🚀 SignFlow / LSA64 — Landmark Çıkarma")
    print("=" * 60)

    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))
    if not video_paths:
        print(f"HATA: Video bulunamadı: {VIDEO_DIR}")
        return
    if args.limit > 0:
        video_paths = video_paths[:args.limit]

    print(f"   Video sayısı: {len(video_paths)}")
    print(f"   Çıktı: {OUTPUT_DIR}")
    print("   Landmarker'lar yükleniyor...")
    pose_lm, hand_lm = build_landmarkers()

    # Split'e göre topla
    buckets = {
        'train': ([], [], []),  # (X, y, rows)
        'val':   ([], [], []),
        'test':  ([], [], []),
    }

    total = len(video_paths)
    for i, path in enumerate(video_paths, 1):
        sign, signer, rep = parse_filename(path)
        label = sign - 1

        if signer in TRAIN_SIGNERS:
            split = 'train'
        elif signer in VAL_SIGNERS:
            split = 'val'
        else:
            split = 'test'

        matrix = extract_from_video(path, pose_lm, hand_lm)
        buckets[split][0].append(matrix)
        buckets[split][1].append(label)
        buckets[split][2].append({
            'path': os.path.relpath(path, project_root),
            'signer': signer, 'rep': rep, 'sign': label,
        })

        if i % 50 == 0 or i == total:
            pct = i / total * 100
            print(f"   İşlenen: {i}/{total} ({pct:.0f}%)")

    # Kaydet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import pandas as pd
    print("\n" + "=" * 60)
    for split, (X, y, rows) in buckets.items():
        if not X:
            print(f"   ⚠️  {split}: örnek yok, atlanıyor.")
            continue
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        np.save(os.path.join(OUTPUT_DIR, f'X_{split}.npy'), X)
        np.save(os.path.join(OUTPUT_DIR, f'y_{split}.npy'), y)
        pd.DataFrame(rows).to_csv(os.path.join(project_root, f'{split}.csv'), index=False)
        print(f"   ✅ {split:5s}: X {X.shape}, y {y.shape}, sınıf: {len(np.unique(y))}")

    print("\n🎉 Landmark çıkarma tamamlandı!")


if __name__ == "__main__":
    main()
