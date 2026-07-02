## Tr

Türk İşaret Dili videolarından el landmark noktalarını çıkararak seçilen işaretleri tanımayı hedefleyen yapay zeka tabanlı bir demo projesidir.

Bu projede videolar önce karelere ayrılır, ardından MediaPipe ile el landmark noktaları çıkarılır. Elde edilen zamansal landmark verileri LSTM/BiLSTM gibi modellerle sınıflandırılarak videodaki işaret tahmin edilir.

## Kullanılan Teknolojiler

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit

## Proje Yapısı

```text
app/          Demo arayüzü
data/         Dataset ve landmark verileri
models/       Eğitilmiş model dosyaları
notebooks/    Deneme ve analiz notebookları
reports/      Rapor ve sonuç dosyaları
src/          Ana Python kodları
```

## Kurulum

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Demo

```bash
streamlit run app/streamlit_app.py
```

---

## En

SignFlow Turkish is an AI-based demo project that aims to recognize selected Turkish Sign Language signs by extracting hand landmark points from videos.

In this project, videos are first split into frames, then hand landmarks are extracted using MediaPipe. The extracted temporal landmark data is classified using models such as LSTM/BiLSTM to predict the sign in the video.

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit

## Project Structure

```text
app/          Demo interface
data/         Dataset and landmark data
models/       Trained model files
notebooks/    Experiment and analysis notebooks
reports/      Reports and result files
src/          Main Python source code
```

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Demo

```bash
streamlit run app/streamlit_app.py
```