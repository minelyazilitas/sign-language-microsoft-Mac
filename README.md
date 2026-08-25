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

Each video is reduced to 30 frames, and each frame gives 225 numbers (hand and
body coordinates) from MediaPipe. A BiLSTM model is trained on these
coordinate sequences to recognize which of 64 signed words is being performed.
`extract.py` turns video into coordinates, `train.py` trains the model,
`predict.py` tests it, and `app.py` / `live.py` are the demos — one using a
saved sample, the other using a live camera.

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

Her video 30 kareye indiriliyor, her kareden de MediaPipe ile 225 sayı (el ve
vücut koordinatları) çıkarılıyor. Bu koordinat dizileriyle eğitilen bir BiLSTM
modeli, 64 kelimeden hangisinin işaret edildiğini tahmin ediyor. `extract.py`
video-koordinat dönüşümünü yapıyor, `train.py` modeli eğitiyor, `predict.py`
test ediyor, `app.py` ve `live.py` da demo kısımları — biri kayıtlı bir
örnekle, diğeri canlı kamerayla çalışıyor.
