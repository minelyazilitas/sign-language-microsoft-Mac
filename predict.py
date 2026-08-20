# Egitilmis modeli test setinde dener ve karisiklik matrisi cikarir.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from labels import label, LABELS, NUM_CLASSES

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "islenmis")
MODEL_PATH = os.path.join(ROOT, "models", "model.keras")


def confusion_matrix(y_true, y_pred):
    m = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def plot_confusion(m):
    out = os.path.join(ROOT, "sonuclar")
    os.makedirs(out, exist_ok=True)
    plt.figure(figsize=(14, 12))
    plt.imshow(m, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), LABELS, rotation=90, fontsize=6)
    plt.yticks(range(NUM_CLASSES), LABELS, fontsize=6)
    plt.xlabel("tahmin")
    plt.ylabel("gercek")
    plt.title("karisiklik matrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion.png"), dpi=120)
    plt.close()


def evaluate():
    model = keras.models.load_model(MODEL_PATH)
    X_test = np.load(os.path.join(DATA, "X_test.npy"))
    y_test = np.load(os.path.join(DATA, "y_test.npy"))

    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = (pred == y_test).mean()
    print(f"test basarisi: {acc*100:.1f}%")

    # ilk 10 ornek
    for i in range(min(10, len(y_test))):
        mark = "dogru" if pred[i] == y_test[i] else "yanlis"
        print(f"  gercek: {label(y_test[i]):12s} tahmin: {label(pred[i]):12s} [{mark}]")

    plot_confusion(confusion_matrix(y_test, pred))
    print("karisiklik matrisi: sonuclar/confusion.png")


def predict_video(video_path):
    from extract import open_models, video_to_array
    model = keras.models.load_model(MODEL_PATH)
    pose, hand = open_models()
    X = video_to_array(video_path, pose, hand)[None, ...]
    prob = model.predict(X, verbose=0)[0]
    i = int(np.argmax(prob))
    return label(i), float(prob[i])


if __name__ == "__main__":
    evaluate()
