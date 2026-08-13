# Bu dosya videoları modelin anlayabileceği sayılara çevirir.
# Her videodan 30 kare seçeriz, her karede MediaPipe ile el ve vücut noktalarını buluruz.
# Bir kare = 75 nokta x 3 eksen (x, y, z) = 225 sayı; tüm video ise (30, 225) olur.
# Sonuçları data/islenmis/ klasörüne X_*.npy ve y_*.npy olarak kaydederiz.

import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(ROOT, "data", "videos")
OUT_DIR = os.path.join(ROOT, "data", "islenmis")
POSE_MODEL = os.path.join(ROOT, "models", "pose_landmarker_lite.task")
HAND_MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")

NUM_FRAMES = 30       # her video 30 kareye sabitlenir
NUM_FEATURES = 225    # 75 nokta x 3 eksen (x, y, z)


def which_set(signer):
    # veriyi kişilere göre ayırıyoruz ki model hiç görmediği kişide sınansın
    # kişi 1-8 eğitim, kişi 9 doğrulama, kişi 10 test
    if signer <= 8:
        return "train"
    if signer == 9:
        return "val"
    return "test"


def open_models():
    # MediaPipe'ın hazır iki modelini açıyoruz: biri vücudu, diğeri elleri bulur
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_hands=2))
    return pose, hand


def frame_to_features(frame_rgb, pose, hand):
    # tek bir kareden 225 sayı çıkarır; bir nokta bulunamazsa yeri 0 kalır
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    body = np.zeros(33 * 3, dtype=np.float32)   # vücut: 33 nokta
    left = np.zeros(21 * 3, dtype=np.float32)   # sol el: 21 nokta
    right = np.zeros(21 * 3, dtype=np.float32)  # sağ el: 21 nokta

    # önce vücudu bul
    p = pose.detect(image)
    if p.pose_landmarks:
        body = np.array([[n.x, n.y, n.z] for n in p.pose_landmarks[0]],
                        dtype=np.float32).flatten()

    # sonra elleri bul; MediaPipe elin sol mu sağ mı olduğunu da söyler
    h = hand.detect(image)
    if h.hand_landmarks:
        for points, side in zip(h.hand_landmarks, h.handedness):
            vec = np.array([[n.x, n.y, n.z] for n in points],
                           dtype=np.float32).flatten()
            if side[0].category_name == "Left":
                left = vec
            else:
                right = vec

    # vücut + sol el + sağ el = 225 sayı
    return np.concatenate([body, left, right])


def video_to_array(path, pose, hand):
    # bütün videoyu (30, 225) boyutunda bir diziye çevirir
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((NUM_FRAMES, NUM_FEATURES), dtype=np.float32)

    # video 30 kareden uzunsa baştan sona eşit aralıklı 30 kare seçeriz
    if total >= NUM_FRAMES:
        picked = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
    else:
        picked = np.arange(total)

    seq = []
    for i in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            seq.append(np.zeros(NUM_FEATURES, dtype=np.float32))
            continue
        # OpenCV kareyi BGR verir, MediaPipe RGB ister; o yüzden çeviriyoruz
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seq.append(frame_to_features(rgb, pose, hand))
    cap.release()

    # video 30 kareden kısaysa kalan yerleri sıfırla doldururuz
    while len(seq) < NUM_FRAMES:
        seq.append(np.zeros(NUM_FEATURES, dtype=np.float32))
    return np.array(seq[:NUM_FRAMES], dtype=np.float32)


def main():
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"{len(videos)} video bulundu")
    pose, hand = open_models()

    # her küme için koordinatları (X) ve etiketleri (y) ayrı ayrı toplarız
    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for n, path in enumerate(videos, 1):
        # dosya adından sınıf, kişi ve tekrar numarasını okuyoruz
        name = os.path.splitext(os.path.basename(path))[0]
        sign, signer, rep = (int(x) for x in name.split("_"))
        s = which_set(signer)
        buckets[s][0].append(video_to_array(path, pose, hand))
        buckets[s][1].append(sign - 1)   # etiket 0-63 arası

        # her 50 videoda bir ilerlemeyi yazdır
        if n % 50 == 0 or n == len(videos):
            print(f"islenen: {n}/{len(videos)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for s, (X, y) in buckets.items():
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        np.save(os.path.join(OUT_DIR, f"X_{s}.npy"), X)
        np.save(os.path.join(OUT_DIR, f"y_{s}.npy"), y)
        print(f"{s}: {X.shape}")

    print("bitti")


if __name__ == "__main__":
    main()
