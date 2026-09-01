import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import numpy as np
import streamlit as st
from tensorflow import keras

from labels import label

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "islenmis")
VIDEO_DIR = os.path.join(ROOT, "data", "videos")
MODEL_PATH = os.path.join(ROOT, "models", "model.keras")
TEST_SIGNER = 10   # test setini bu kişiden ayırmıştık

st.set_page_config(page_title="SignFlow")


@st.cache_resource
def load_model():
    # modeli bir kere yükleyip hafızada tutuyoruz ki her tıklamada tekrar yüklenmesin
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


@st.cache_data
def test_videos():
    # extract.py videoları sıralı okuyup test setine kişi 10'u koymuştu,
    # o yüzden test örneğinin sırası ile video dosyasının sırası aynı
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    return [v for v in videos
            if int(os.path.basename(v).split("_")[1]) == TEST_SIGNER]


st.title("SignFlow - İşaret Dili Tanıma")

model = load_model()
if model is None:
    st.error("Önce modeli eğit: python train.py")
    st.stop()

X_test = np.load(os.path.join(DATA, "X_test.npy"))
y_test = np.load(os.path.join(DATA, "y_test.npy"))

# kullanıcı test setinden bir örnek seçer
i = st.number_input("Test örneğinin numarası", 0, len(X_test) - 1, 0)
truth = label(int(y_test[i]))

# videoyu gösteriyoruz ki izleyen kişi işareti görebilsin
videos = test_videos()
if i < len(videos):
    st.video(videos[i])
else:
    st.info("video bulunamadı, sadece tahmin gösterilecek")

if st.button("Tahmin et"):
    # seçilen örneği modele verip en yüksek olasılıklı kelimeyi bulur
    prob = model.predict(X_test[i:i + 1], verbose=0)[0]
    pred = int(np.argmax(prob))
    conf = prob[pred] * 100
    correct = pred == y_test[i]

    title = "Doğru tahmin" if correct else "Yanlış tahmin"
    st.markdown(f"### {title}: {label(pred)} (%{conf:.1f})")
    st.write(f"Gerçek kelime: **{truth}**")
