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
POSE_MODEL = os.path.join(ROOT, "models", "pose_landmarker_heavy.task")
HAND_MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")

NUM_FRAMES = 30
THRESHOLD = 0.3      # tespit esigi
CROP_RADIUS = 120     # bilek etrafinda kac piksel kirpilacak

POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
HAND_WRIST = 0
HAND_MIDDLE_BASE = 9   # el buyuklugu icin referans nokta

# toplam ozellik: konum(234) + hiz(234) + ivme(234) + maske(3)
POS_SIZE = 234
NUM_FEATURES = POS_SIZE * 3 + 3


def which_set(signer):
    # kisilere gore ayiriyoruz ki model test ettigimiz kisiyi hic gormemis olsun
    if signer <= 8:
        return "train"
    if signer == 9:
        return "val"
    return "test"


def open_models():
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=THRESHOLD))
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=THRESHOLD))
    return pose, hand


def crop_and_retry(rgb, hand, wrist_x, wrist_y):
    H, W, _ = rgb.shape
    r = CROP_RADIUS
    x1, y1 = max(0, int(wrist_x - r)), max(0, int(wrist_y - r))
    x2, y2 = min(W, int(wrist_x + r)), min(H, int(wrist_y + r))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    crop = rgb[y1:y2, x1:x2]
    crop_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = hand.detect(crop_img)
    if not result.hand_landmarks:
        return None

    cw, ch = x2 - x1, y2 - y1
    for landmarks, side in zip(result.hand_landmarks, result.handedness):
        pts = np.array([[x1 + n.x * cw, y1 + n.y * ch] for n in landmarks], dtype=np.float32)
        pts[:, 0] /= W
        pts[:, 1] /= H
        return pts, side[0].category_name
    return None


def frame_raw(rgb, pose, hand):
    # bir kareden ham pose ve el noktalarini cikarir
    H, W, _ = rgb.shape
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_xy = np.zeros((33, 2), dtype=np.float32)
    left_xy = np.zeros((21, 2), dtype=np.float32)
    right_xy = np.zeros((21, 2), dtype=np.float32)
    pose_found = False

    p = pose.detect(image)
    if p.pose_landmarks:
        lm = p.pose_landmarks[0]
        pose_xy = np.array([[n.x, n.y] for n in lm], dtype=np.float32)
        pose_found = True

    h = hand.detect(image)
    found = {}
    if h.hand_landmarks:
        for landmarks, side in zip(h.hand_landmarks, h.handedness):
            pts = np.array([[n.x, n.y] for n in landmarks], dtype=np.float32)
            found[side[0].category_name] = pts

    # eksik el varsa, o elin bilegi etrafini kirpip tekrar deniyoruz
    if pose_found:
        if "Left" not in found:
            result = crop_and_retry(rgb, hand, pose_xy[POSE_LEFT_WRIST, 0] * W,
                                    pose_xy[POSE_LEFT_WRIST, 1] * H)
            if result:
                found[result[1]] = result[0]
        if "Right" not in found:
            result = crop_and_retry(rgb, hand, pose_xy[POSE_RIGHT_WRIST, 0] * W,
                                    pose_xy[POSE_RIGHT_WRIST, 1] * H)
            if result:
                found[result[1]] = result[0]

    left_found = "Left" in found
    right_found = "Right" in found
    if left_found:
        left_xy = found["Left"]
    if right_found:
        right_xy = found["Right"]

    return pose_xy, left_xy, right_xy, pose_found, left_found, right_found


def normalize_frame(pose_xy, left_xy, right_xy, pose_found, left_found, right_found):
    # Omuz ortasini bulup omuz genisligine boluyoruz ki kisi kameraya yakin da dursa uzak da ayni gorunsun
    if not pose_found:
        global_pos = np.zeros(150, dtype=np.float32)
        local_pos = np.zeros(84, dtype=np.float32)
        return global_pos, local_pos

    left_shoulder, right_shoulder = pose_xy[POSE_LEFT_SHOULDER], pose_xy[POSE_RIGHT_SHOULDER]
    center = (left_shoulder + right_shoulder) / 2
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale < 1e-3:
        scale = 1.0

    angle = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)

    def shift_scale_rotate(pts):
        shifted = (pts - center) / scale
        return shifted @ rotation.T

    pose_n = shift_scale_rotate(pose_xy)
    left_n = shift_scale_rotate(left_xy) if left_found else np.zeros((21, 2), dtype=np.float32)
    right_n = shift_scale_rotate(right_xy) if right_found else np.zeros((21, 2), dtype=np.float32)

    global_pos = np.concatenate([pose_n.flatten(), left_n.flatten(), right_n.flatten()])

    # elin noktalarini kendi bilegine ve kendi buyuklugune gore yeniden hesapliyoruz
    def hand_local(pts, found):
        if not found:
            return np.zeros(42, dtype=np.float32)
        wrist = pts[HAND_WRIST]
        size = np.linalg.norm(pts[HAND_MIDDLE_BASE] - wrist)
        if size < 1e-3:
            size = 1.0
        return ((pts - wrist) / size).flatten()

    local_pos = np.concatenate([hand_local(left_xy, left_found), hand_local(right_xy, right_found)])

    return global_pos.astype(np.float32), local_pos.astype(np.float32)


def video_to_array(path, pose, hand):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((NUM_FRAMES, NUM_FEATURES), dtype=np.float32)

    if total >= NUM_FRAMES:
        picked = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
    else:
        picked = np.arange(total)

    pos_list = []
    mask_list = []
    for i in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            pos_list.append(np.zeros(POS_SIZE, dtype=np.float32))
            mask_list.append(np.zeros(3, dtype=np.float32))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_xy, left_xy, right_xy, pose_found, left_found, right_found = frame_raw(rgb, pose, hand)
        global_pos, local_pos = normalize_frame(pose_xy, left_xy, right_xy, pose_found, left_found, right_found)
        pos_list.append(np.concatenate([global_pos, local_pos]))
        mask_list.append(np.array([pose_found, left_found, right_found], dtype=np.float32))
    cap.release()

    while len(pos_list) < NUM_FRAMES:
        pos_list.append(np.zeros(POS_SIZE, dtype=np.float32))
        mask_list.append(np.zeros(3, dtype=np.float32))

    pos = np.array(pos_list[:NUM_FRAMES], dtype=np.float32)
    mask = np.array(mask_list[:NUM_FRAMES], dtype=np.float32)

    # hiz = konum farki, ivme = hiz farki (ikisi de bir onceki kareye gore)
    velocity = np.zeros_like(pos)
    velocity[1:] = pos[1:] - pos[:-1]
    acceleration = np.zeros_like(pos)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # hicbir sey bulunamayan karelerde hiz/ivme de sifir olsun (maskeleme calısabilsin diye)
    empty = mask.sum(axis=1) == 0
    velocity[empty] = 0
    acceleration[empty] = 0

    return np.concatenate([pos, velocity, acceleration, mask], axis=1).astype(np.float32)


def main():
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"{len(videos)} video bulundu")
    pose, hand = open_models()

    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for n, path in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        sign, signer, rep = (int(x) for x in name.split("_"))
        s = which_set(signer)
        buckets[s][0].append(video_to_array(path, pose, hand))
        buckets[s][1].append(sign - 1)

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
