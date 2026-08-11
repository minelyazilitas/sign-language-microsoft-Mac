"""
SignFlow — Streamlit Demo Uygulaması
Türk İşaret Dili tahmin arayüzü.
"""

import os
import sys
import numpy as np

# Proje kök dizinini ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# TensorFlow uyarılarını azalt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
from tensorflow import keras

from src.utils import ID_TO_WORD, NUM_CLASSES


# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="SignFlow — Türk İşaret Dili Tanıma",
    page_icon="🤟",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Eğitilmiş modeli yükle ve önbelleğe al."""
    model_path = os.path.join(project_root, 'models', 'best_model.keras')
    if not os.path.exists(model_path):
        return None
    return keras.models.load_model(model_path)


def main():
    # --- Başlık ---
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🤟 SignFlow</h1>
        <p style="font-size: 1.2rem; color: #888;">
            Türk İşaret Dili Tanıma Sistemi — BiLSTM Tabanlı Demo
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # --- Model Yükleme ---
    model = load_model()
    
    if model is None:
        st.error(
            "⚠️ Eğitilmiş model bulunamadı! "
            "Lütfen önce `python src/train.py` komutunu çalıştırın."
        )
        st.stop()
    
    st.success("✅ Model başarıyla yüklendi!")

    # --- İki Mod ---
    tab1, tab2 = st.tabs(["📁 Test Setinden Seç", "📊 Model Bilgileri"])
    
    with tab1:
        st.subheader("Test Setinden Örnek Seç ve Tahmin Yap")

        # Önceden işlenmiş test verisini yükle (X_test.npy / y_test.npy)
        import numpy as _np
        x_test_path = os.path.join(project_root, 'data', 'processed', 'X_test.npy')
        y_test_path = os.path.join(project_root, 'data', 'processed', 'y_test.npy')

        if not (os.path.exists(x_test_path) and os.path.exists(y_test_path)):
            st.warning(
                "İşlenmiş test verisi bulunamadı. "
                "Önce `python src/extract_landmarks.py` komutunu çalıştırın."
            )
            st.stop()

        X_test = _np.load(x_test_path)
        y_test = _np.load(y_test_path)

        # Kullanıcı bir örnek seçsin
        col1, col2 = st.columns([1, 2])

        with col1:
            sample_idx = st.number_input(
                "Örnek indeksi seçin",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                step=1,
            )

            true_label_id = int(y_test[sample_idx])
            true_word = ID_TO_WORD.get(true_label_id, f"ID: {true_label_id}")

            st.markdown(f"""
            **Seçilen Kayıt:**
            - 🔢 İndeks: `{sample_idx}` / {len(X_test) - 1}
            - 🏷️ Gerçek Etiket: **{true_word}** (ID: {true_label_id})
            - 📐 Girdi: 30 × 225 landmark dizisi
            """)

            predict_btn = st.button("🔮 Tahmin Yap", type="primary", use_container_width=True)

        with col2:
            if predict_btn:
                with st.spinner("Tahmin yapılıyor..."):
                    X_input = X_test[sample_idx:sample_idx + 1]  # (1, 30, 225)
                    probs = model.predict(X_input, verbose=0)[0]
                    pred_id = int(np.argmax(probs))
                    confidence = float(probs[pred_id]) * 100
                    pred_word = ID_TO_WORD.get(pred_id, f"ID: {pred_id}")

                # Sonuç göster
                is_correct = pred_id == true_label_id

                if is_correct:
                    st.success(f"✅ Doğru Tahmin!")
                else:
                    st.error(f"❌ Yanlış Tahmin")

                st.markdown(f"""
                ### 🔮 Tahmin Sonucu

                | | Değer |
                |---|---|
                | **Tahmin** | **{pred_word}** |
                | **Güven** | **{confidence:.1f}%** |
                | **Gerçek** | {true_word} |
                """)

                # Top-5 tahminler
                st.markdown("#### 📊 En Yüksek 5 Tahmin")
                top5_idx = np.argsort(probs)[::-1][:5]

                chart_data = {
                    "Kelime": [ID_TO_WORD.get(i, f"ID:{i}") for i in top5_idx],
                    "Güven (%)": [float(probs[i]) * 100 for i in top5_idx],
                }
                st.bar_chart(
                    pd.DataFrame(chart_data).set_index("Kelime"),
                    horizontal=True,
                )
    
    with tab2:
        st.subheader("Model Mimarisi ve Bilgileri")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sınıf Sayısı", NUM_CLASSES)
        col2.metric("Girdi Boyutu", "30 × 225")
        col3.metric("Model Tipi", "BiLSTM")
        
        st.markdown("#### 📐 Model Özeti")
        
        # Model summary'yi string olarak al
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        st.code("\n".join(summary_lines), language="text")
        
        # Eğitim raporu varsa göster
        report_path = os.path.join(project_root, 'reports', 'training_history.png')
        if os.path.exists(report_path):
            st.markdown("#### 📈 Eğitim Grafikleri")
            st.image(report_path)
        
        cm_path = os.path.join(project_root, 'reports', 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.markdown("#### 🗂️ Confusion Matrix")
            st.image(cm_path)


if __name__ == "__main__":
    main()
