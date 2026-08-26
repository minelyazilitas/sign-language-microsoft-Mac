import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from labels import NUM_CLASSES

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "islenmis")
MODEL_PATH = os.path.join(ROOT, "models", "model.keras")


def time_warp(X):
    # zaman penceresini rastgele degistirip ayni isareti farkli ritimde gosteriyoruz
    n, frames, _ = X.shape
    out = np.zeros_like(X)
    old_t = np.arange(frames)

    for i in range(n):
        span = np.random.uniform(0.8, 1.0) * (frames - 1)
        start = np.random.uniform(0, (frames - 1) - span)
        new_t = np.linspace(start, start + span, frames)

        # konum sutunlari
        for c in range(234):
            out[i, :, c] = np.interp(new_t, old_t, X[i, :, c])
        # maske sutunlari, 0/1 kalsin diye yuvarliyoruz
        for c in range(702, 705):
            out[i, :, c] = np.round(np.interp(new_t, old_t, X[i, :, c]))

    # kareler kaydigi icin eski hiz/ivme degerlerini bastan hesapliyoruz
    pos = out[:, :, 0:234]
    vel = np.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    acc = np.zeros_like(pos)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
    out[:, :, 234:468] = vel
    out[:, :, 468:702] = acc

    return out


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


def plot_history(history):
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
    plt.savefig(os.path.join(out, "egitim.png"), dpi=120)
    plt.close()


def main():
    X_train = np.load(os.path.join(DATA, "X_train.npy"))
    y_train = np.load(os.path.join(DATA, "y_train.npy"))
    X_val = np.load(os.path.join(DATA, "X_val.npy"))
    y_val = np.load(os.path.join(DATA, "y_val.npy"))

    # Ritmi degistirilmis bir kopya daha ekleyip egitim setini ikiye katliyoruz
    X_warped = time_warp(X_train)
    noise = np.random.normal(0, 0.02, X_warped.shape).astype(np.float32) * (X_warped != 0)
    X_train = np.concatenate([X_train, X_warped + noise])
    y_train = np.concatenate([y_train, y_train])
    print(f"egitim: {X_train.shape}, dogrulama: {X_val.shape}")

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # en iyi modeli kaydeder, gelisme durursa erken durur (gereksiz ısınma/overfitting önleyici)
    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                                        save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=6, min_lr=1e-6),
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, callbacks=callbacks)

    plot_history(history)
    print(f"en iyi dogrulama: {max(history.history['val_accuracy']):.3f}")
    print(f"model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()
