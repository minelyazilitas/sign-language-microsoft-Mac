import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import json
import sys
import argparse
import numpy as np
from tensorflow import keras

from train import SEED, build_model, time_warp, add_noise

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data(data_name):
    data_dir = os.path.join(ROOT, "data", data_name)
    X = np.concatenate([np.load(os.path.join(data_dir, f"X_{s}.npy"))
                        for s in ["train", "val", "test"]])
    y = np.concatenate([np.load(os.path.join(data_dir, f"y_{s}.npy"))
                        for s in ["train", "val", "test"]])

    # extract.py videoları sıralı okuyup train/val/test diye ayırıyor,
    # aynı sırayı burada da kurup her örneğin kimden geldiğini buluyoruz
    videos = sorted(glob.glob(os.path.join(ROOT, "data", "videos", "*.mp4")))
    signers = np.array([int(os.path.basename(v).split("_")[1]) for v in videos])
    signers = np.concatenate([signers[signers <= 8], signers[signers == 9],
                              signers[signers == 10]])

    if len(X) != len(signers):
        raise ValueError("video sayisi ile veri sayisi ayni degil")
    return X, y, signers


def test_one_signer(X, y, signers, test_signer, model_path):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    val_signer = test_signer % 10 + 1
    train = (signers != test_signer) & (signers != val_signer)
    val = signers == val_signer
    test = signers == test_signer

    X_train = X[train]
    X_aug = add_noise(time_warp(X_train.copy()))
    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y[train], y[train]])

    model = build_model((X.shape[1], X.shape[2]))
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                        save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=6, min_lr=1e-6),
    ]
    history = model.fit(X_train, y_train, validation_data=(X[val], y[val]),
                        epochs=100, batch_size=32, callbacks=callbacks, verbose=0)
    # testi son epoch ile değil, kaydedilen en iyi modelle yapıyoruz
    best_model = keras.models.load_model(model_path)
    pred = np.argmax(best_model.predict(X[test], verbose=0), axis=1)
    score = float((pred == y[test]).mean() * 100)
    return val_signer, score, len(history.history["loss"])


def main():
    p = argparse.ArgumentParser(description="10 kisiyle capraz dogrulama yapar")
    p.add_argument("--veri", default="islenmis", help="data/ altindaki veri klasoru")
    p.add_argument("--ad", default=None, help="sonuc dosyasi adi (verilmezse veri adi kullanilir)")
    a = p.parse_args()

    if a.ad is None:
        a.ad = f"capraz_{a.veri}"
    report_path = os.path.join(ROOT, "sonuclar", f"{a.ad}.json")
    model_dir = os.path.join(ROOT, "models", a.ad)
    if os.path.exists(report_path) or os.path.exists(model_dir):
        sys.exit(f"bu ad zaten kullanilmis: {a.ad}\nbaska ad ver: --ad yeni_ad")

    X, y, signers = load_data(a.veri)
    os.makedirs(model_dir)
    results = []
    for test_signer in range(1, 11):
        model_path = os.path.join(model_dir, f"kisi_{test_signer}.keras")
        val_signer, score, epochs = test_one_signer(X, y, signers, test_signer, model_path)
        results.append({"test_kisi": test_signer, "val_kisi": val_signer,
                        "test_dogrulugu": round(score, 2), "epoch": epochs})
        print(f"test kisi {test_signer}, val kisi {val_signer}: %{score:.2f}")

    scores = np.array([r["test_dogrulugu"] for r in results])
    report = {"veri": a.veri, "seed": SEED, "sonuclar": results,
              "ortalama": round(float(scores.mean()), 2),
              # slayttaki standart sapmalarla aynı hesap, 10 kişi bir örneklem
              "standart_sapma": round(float(scores.std(ddof=1)), 2)}
    os.makedirs(os.path.join(ROOT, "sonuclar"), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nortalama: %{report['ortalama']:.2f} ± {report['standart_sapma']:.2f}")
    print(f"rapor: {report_path}")


if __name__ == "__main__":
    main()
