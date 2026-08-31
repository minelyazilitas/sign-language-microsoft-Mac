import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import shutil
import datetime
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from labels import NUM_CLASSES

ROOT = os.path.dirname(os.path.abspath(__file__))

SEED = 42
keras.utils.set_random_seed(SEED)

PARTS = [
    (702, [(0, 66)]),                  # govde
    (703, [(66, 108), (150, 192)]),    # sol el
    (704, [(108, 150), (192, 234)]),   # sag el
]


def tidy(X):
    for mask_col, ranges in PARTS:
        yok = X[:, :, mask_col] == 0
        for lo, hi in ranges:
            X[:, :, lo:hi][yok] = 0

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
    return X


def time_warp(X):
    # zaman penceresini rastgele değitirip aynı işareti farklı ritimde gösteriyoruz
    n, frames, _ = X.shape
    out = np.zeros_like(X)
    old_t = np.arange(frames)

    for i in range(n):
        span = np.random.uniform(0.8, 1.0) * (frames - 1)
        start = np.random.uniform(0, (frames - 1) - span)
        new_t = np.linspace(start, start + span, frames)

        # Maskeyi en yakın eski kareden alıyoruz.
        nearest = np.rint(new_t).astype(int)
        out[i, :, 702:705] = X[i, nearest, 702:705]

        # Her parcanin konumunu sadece o parçanın göründüğü karelerden alıyoruz.

        for mask_col, ranges in PARTS:
            visible = np.where(X[i, :, mask_col] == 1)[0]
            if len(visible) == 0:
                continue
            for lo, hi in ranges:
                for c in range(lo, hi):
                    out[i, :, c] = np.interp(new_t, old_t[visible], X[i, visible, c])

    return tidy(out)


def add_noise(X):
    pos = X[:, :, 0:234]
    noise = np.random.normal(0, 0.02, pos.shape).astype(np.float32)
    X[:, :, 0:234] = pos + noise * (pos != 0)
    return tidy(X)


def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(160, return_sequences=True, dropout=0.3)),
        layers.Bidirectional(layers.LSTM(96, dropout=0.3)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def write_meta(path, name, data_dir, X_train, X_val, best_val, epochs):
    meta = {
        "deney": name,
        "tarih": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "veri_klasoru": os.path.basename(data_dir),
        "egitim_ornegi": int(len(X_train)),
        "augmentasyonla_egitim_ornegi": int(len(X_train)) * 2,
        "dogrulama_ornegi": int(len(X_val)),
        "ozellik_sayisi": int(X_train.shape[-1]),
        "seed": SEED,
        "augmentasyon": "time_warp + add_noise (egitim seti ikiye katlaniyor)",
        "epoch": epochs,
        "batch": 32,
        "en_iyi_val_accuracy": round(float(best_val), 4),
        "not": "test skoru icin: python predict.py",
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def parse_args():
    p = argparse.ArgumentParser(description="isaret dili modelini egitir")
    p.add_argument("ad", nargs="?", default=None,
                   help="deney adi, ornek: zaman_aug. verilmezse tarih kullanilir")
    p.add_argument("--veri", default="islenmis",
                   help="data/ altindaki veri klasoru (varsayilan: islenmis)")
    a = p.parse_args()
    if a.ad is None:
        a.ad = datetime.datetime.now().strftime("%m%d_%H%M")
    return a


def plot_history(history, name):
    out = os.path.join(ROOT, "sonuclar")
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["loss"], label="egitim")
    ax[0].plot(history.history["val_loss"], label="dogrulama")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="egitim")
    ax[1].plot(history.history["val_accuracy"], label="dogrulama")
    ax[1].set_title("accuracy"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"egitim_{name}.png"), dpi=120)
    plt.close()


def main():
    args = parse_args()
    data_dir = os.path.join(ROOT, "data", args.veri)
    model_path = os.path.join(ROOT, "models", f"model_{args.ad}.keras")
    meta_path = os.path.join(ROOT, "models", f"model_{args.ad}.json")

    if not os.path.isdir(data_dir):
        sys.exit(f"veri klasoru yok: {data_dir}")

    # Eskiden her eğitim models/model.keras üzerine yazıyordu ve önceki deney
    # kayboluyordu. Artık aynı adla ikinci kez eğitim yapılmasına izin vermiyoruz.
    if os.path.exists(model_path):
        sys.exit(f"bu adda model zaten var: {model_path}\n"
                 f"baska bir deney adi ver: python train.py <ad>")

    print(f"deney: {args.ad}")
    print(f"veri : {data_dir}")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    raw_train = X_train

    # Ritmi değiştirilmiş bir kopya daha ekleyip egitim setini ikiye katlıyoruz
    X_warped = add_noise(time_warp(X_train))
    X_train = np.concatenate([X_train, X_warped])
    y_train = np.concatenate([y_train, y_train])
    print(f"egitim: {X_train.shape}, dogrulama: {X_val.shape}")

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # en iyi modeli kaydediyoruz, gelişme durursa erkenden durur (gereksiz ısınma/overfitting önleyici)
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                        save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=6, min_lr=1e-6),
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, callbacks=callbacks)

    plot_history(history, args.ad)
    best_val = max(history.history["val_accuracy"])
    write_meta(meta_path, args.ad, data_dir, raw_train, X_val,
               best_val, len(history.history["loss"]))

    shutil.copy(model_path, os.path.join(ROOT, "models", "model.keras"))

    print(f"en iyi dogrulama: {best_val:.3f}")
    print(f"model kaydedildi: {model_path}")
    print(f"kunye kaydedildi: {meta_path}")


if __name__ == "__main__":
    main()
