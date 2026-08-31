# SignFlow

🇺🇸 English

## Purpose

Before this project, I had built a small letter-recognition system using a
dataset I put together myself. This project takes that a step further:
recognizing whole words in sign language instead of single letters, with the
goal of eventually working towards full sentences.

## Technologies

- Python
- OpenCV — reading video and camera input
- MediaPipe — finding hand and body landmarks
- TensorFlow / Keras — the BiLSTM model
- NumPy
- Streamlit — demo interface

## How it works

Each video is reduced to 30 frames. Each frame has 705 numbers: hand and body
coordinates, their movement between frames, and simple "found / not found"
markers. A BiLSTM model is trained on these
coordinate sequences to recognize which of 64 signed words is being performed.
`extract.py` turns video into coordinates, `train.py` trains the model,
`predict.py` tests it, and `app.py` / `live.py` are the demos — one using a
saved sample, the other using a live camera.

The split is by signer, not random: signers 1–8 train, 9 validates, 10 tests.
The model never sees the test person during training. `cross_validation.py`
repeats this for all ten signers so every person gets tested once, and
`analyze.py` compares those scores against the hand detection rate.

## Setup

The videos and the MediaPipe model files are not in this repository — they are
too large for git. Download them first:

**1. Dataset (LSA64):** https://facundoq.github.io/datasets/lsa64/
Download the "cut" version and put the 3200 `.mp4` files into `data/videos/`.
File names are `class_signer_repeat`, for example `033_001_001.mp4`.

**2. MediaPipe models:** put both files into `models/`.

```bash
mkdir -p models data/videos
curl -o models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**3. Libraries:**

```bash
pip install -r requirements.txt
```

## Run order

```bash
python extract.py --hedef islenmis_ham
python fill_gaps.py --kaynak islenmis_ham --hedef islenmis
python train.py ilk_deney --veri islenmis
python predict.py --veri islenmis --model model_ilk_deney.keras
python cross_validation.py --veri islenmis --ad capraz_islenmis
python analyze.py --rapor capraz_islenmis.json
```

Extraction takes hours, cross validation about an hour. The demos need a
trained model:

```bash
streamlit run app.py     # browser interface, uses a saved sample
python live.py           # live camera, press q to quit
```

Every step writes to a **new** folder and refuses to overwrite an existing one.
Each data folder gets a `kunye.json` (where it came from, settings, hand
detection rate) and each model gets a `model_<name>.json` next to it (which data
version, seed, augmentation, best validation score). This is deliberate: earlier
every step wrote over `data/islenmis` and every run wrote over `models/model.keras`,
so it became impossible to tell which model was trained on which data.

---

🇹🇷 Türkçe

## Amaç

Bu projeden önce, kendi oluşturduğum bir veri setiyle harfleri tanıyan küçük
bir sistem yapmıştım. Bu proje onu bir adım öteye taşıyor: tek tek harfler
yerine kelimeleri tanımak, ileride de cümlelere doğru genişletmek hedefiyle.

## Kullanılan teknolojiler

- Python
- OpenCV — video ve kamera okuma
- MediaPipe — el ve vücut noktalarını bulma
- TensorFlow / Keras — BiLSTM modeli
- NumPy
- Streamlit — demo arayüzü

## Nasıl çalışır

Her video 30 kareye indiriliyor. Her karede 705 sayı var: el ve vücut
koordinatları, kareler arasındaki hareket bilgisi ve basit "bulundu / bulunmadı"
işaretleri. Bu dizilerle eğitilen BiLSTM
modeli, 64 kelimeden hangisinin işaret edildiğini tahmin ediyor. `extract.py`
video-koordinat dönüşümünü yapıyor, `train.py` modeli eğitiyor, `predict.py`
test ediyor, `app.py` ve `live.py` da demo kısımları — biri kayıtlı bir
örnekle, diğeri canlı kamerayla çalışıyor.

Veri kişiye göre ayrılıyor, rastgele değil: 1-8 arası kişiler eğitim, 9
doğrulama, 10 test. Model test edilen kişiyi eğitim sırasında hiç görmüyor.
`cross_validation.py` bunu on kişinin hepsi için tekrarlıyor, böylece herkes
bir kez test ediliyor; `analyze.py` de çıkan skorları el tespit oranıyla
karşılaştırıyor.

## Kurulum

Videolar ve MediaPipe model dosyaları bu depoda yok, git için fazla büyükler.
Önce onları indirmen gerekiyor:

**1. Veri seti (LSA64):** https://facundoq.github.io/datasets/lsa64/
"Cut" sürümünü indirip 3200 `.mp4` dosyasını `data/videos/` içine koy.
Dosya adları `sınıf_kişi_tekrar` biçiminde, örnek: `033_001_001.mp4`.

**2. MediaPipe modelleri:** ikisini de `models/` içine koy.

```bash
mkdir -p models data/videos
curl -o models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**3. Kütüphaneler:**

```bash
pip install -r requirements.txt
```

## Çalıştırma sırası

```bash
python extract.py --hedef islenmis_ham
python fill_gaps.py --kaynak islenmis_ham --hedef islenmis
python train.py ilk_deney --veri islenmis
python predict.py --veri islenmis --model model_ilk_deney.keras
python cross_validation.py --veri islenmis --ad capraz_islenmis
python analyze.py --rapor capraz_islenmis.json
```

Çıkarım saatler sürüyor, çapraz doğrulama da yaklaşık bir saat. Demolar için
eğitilmiş bir model gerekiyor:

```bash
streamlit run app.py     # tarayıcıda arayüz, kayıtlı bir örnekle çalışır
python live.py           # canlı kamera, çıkmak için q
```

Her adım **yeni** bir klasöre yazıyor, var olanın üstüne yazmayı reddediyor.
Her veri klasörünün yanına `kunye.json` (nereden geldiği, ayarlar, el tespit
oranı), her modelin yanına da `model_<ad>.json` (hangi veriyle eğitildiği, seed,
augmentasyon, en iyi doğrulama skoru) yazılıyor. Bu bilerek böyle: önceden her
adım `data/islenmis` üstüne, her eğitim de `models/model.keras` üstüne yazıyordu
ve bir süre sonra hangi modelin hangi veriyle eğitildiği anlaşılmaz oldu.
