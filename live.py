# son 30 kareyi biriktirir, modele sorup ekranın altına tahmini yazar.
# Çıkmak için q tuşuna basın.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import collections
import time
import cv2
import numpy as np
from tensorflow import keras

from extract import open_models, frame_raw, normalize_frame, NUM_FRAMES
from labels import label

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "model.keras")
WAIT_TIME = 2.0    # bekleme süresi ekledik
FRAME_SKIP = 2     
MAX_GAP = 3        

HANDS = {
    1: [(66, 108), (150, 192)],    # sol el
    2: [(108, 150), (192, 234)],   # sag el
}


def fill_gaps(pos, mask):
    # eğitim verisinde kısa boşlukları doldurmuştuk, model onu bekliyor.
    # burada da aynı şeyi yapmazsak canlı görüntü eğitimdekinden farklı gelir
    for mask_col, ranges in HANDS.items():
        present = np.where(mask[:, mask_col] == 1)[0]
        if len(present) < 2:
            continue
        for a, b in zip(present[:-1], present[1:]):
            gap = b - a - 1
            if gap == 0 or gap > MAX_GAP:
                continue
            for t in range(a + 1, b):
                w = (t - a) / (b - a)   # a’ya mı b’ye mi daha yakın
                for lo, hi in ranges:
                    pos[t, lo:hi] = pos[a, lo:hi] * (1 - w) + pos[b, lo:hi] * w
                mask[t, mask_col] = 1
    return pos, mask


def build_sequence(pos_buffer, mask_buffer):
    # eğitimdeki gibi konum + hız + ivme + maske birleştirir
    pos = np.array(pos_buffer, dtype=np.float32)
    mask = np.array(mask_buffer, dtype=np.float32)
    # hız/ivme hesaplamadan önce boşlukları dolduruyoruz, eğitimdeki sırayla aynı
    pos, mask = fill_gaps(pos, mask)
    velocity = np.zeros_like(pos)
    velocity[1:] = pos[1:] - pos[:-1]
    acceleration = np.zeros_like(pos)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # hiçbir şey bulunamayan karelerde hız/ivme de sıfır olsun 
    empty = mask.sum(axis=1) == 0
    velocity[empty] = 0
    acceleration[empty] = 0

    return np.concatenate([pos, velocity, acceleration, mask], axis=1)


def open_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"kamera {i} kullaniliyor")
                return cap
        cap.release()
    return None


def main():
    model = keras.models.load_model(MODEL_PATH)
    pose, hand = open_models()

    cap = open_camera()
    if cap is None:
        print("kamera acilamadi")
        print("Sistem Ayarlari > Gizlilik ve Guvenlik > Kamera icinden")
        print("terminale izin verip uygulamayi yeniden baslatman gerekebilir")
        return

    pos_buffer = collections.deque(maxlen=NUM_FRAMES)
    mask_buffer = collections.deque(maxlen=NUM_FRAMES)
    text = "bekleniyor..."
    last_predict_time = 0
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_xy, left_xy, right_xy, pose_found, left_found, right_found = frame_raw(rgb, pose, hand)
        global_pos, local_pos = normalize_frame(pose_xy, left_xy, right_xy, pose_found, left_found, right_found)

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            pos_buffer.append(np.concatenate([global_pos, local_pos]))
            mask_buffer.append([pose_found, left_found, right_found])

        # el ve omuzlar görünmeden tahmin yaptırmıyoruz
        hands_visible = left_found or right_found
        time_up = time.time() - last_predict_time >= WAIT_TIME

        if not (hands_visible and pose_found):
            text = "bekleniyor..."
        elif len(pos_buffer) == NUM_FRAMES and time_up:
            seq = build_sequence(pos_buffer, mask_buffer)
            prob = model.predict(seq[None, ...], verbose=0)[0]
            pred = int(np.argmax(prob))
            conf = prob[pred] * 100
            text = f"{label(pred)} (%{conf:.0f})"
            last_predict_time = time.time()

        font_scale = 1.8
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
        x = (frame.shape[1] - text_w) // 2
        y = frame.shape[0] - 30
        shift = 2
        for dx in (-shift, 0, shift):
            for dy in (-shift, 0, shift):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(frame, text, (x + dx, y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("SignFlow - canli tahmin", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
