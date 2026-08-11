"""
SignFlow — Tahmin ve Değerlendirme
Eğitilmiş modeli test seti üzerinde değerlendirir.
"""

import os
import sys
import numpy as np

# Proje kök dizinini ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import (
    load_processed_data,
    plot_confusion_matrix,
    print_classification_report,
    ID_TO_WORD,
    NUM_CLASSES,
)

# TensorFlow uyarılarını azalt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras


def evaluate_model():
    """Test seti üzerinde modeli değerlendirir."""
    print("🔍 SignFlow — Model Değerlendirmesi")
    print("="*60)
    
    # --- 1. Modeli Yükle ---
    model_path = os.path.join(project_root, 'models', 'best_model.keras')
    
    if not os.path.exists(model_path):
        print(f"HATA: Eğitilmiş model bulunamadı: {model_path}")
        print("Önce 'python src/train.py' komutunu çalıştırın.")
        return
    
    print(f"\n📂 Model yükleniyor: {model_path}")
    model = keras.models.load_model(model_path)
    
    # --- 2. Test Verisini Yükle ---
    print("\n📂 Test verisi yükleniyor...")
    X_test, y_test = load_processed_data('test')
    print(f"   Test seti: {X_test.shape[0]} örnek")
    
    # --- 3. Tahmin Yap ---
    print("\n🎯 Tahminler yapılıyor...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # --- 4. Genel Metrikler ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*60}")
    print(f"  📊 TEST SONUÇLARI")
    print(f"{'='*60}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # --- 5. Detaylı Rapor ve Görseller ---
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    
    # --- 6. Örnek Tahminler ---
    print(f"\n{'='*60}")
    print(f"  🔎 ÖRNEK TAHMİNLER (İlk 10)")
    print(f"{'='*60}")
    
    for i in range(min(10, len(y_test))):
        true_label = ID_TO_WORD.get(y_test[i], f"ID:{y_test[i]}")
        pred_label = ID_TO_WORD.get(y_pred[i], f"ID:{y_pred[i]}")
        confidence = y_pred_probs[i][y_pred[i]] * 100
        status = "✅" if y_test[i] == y_pred[i] else "❌"
        print(f"  {status} Gerçek: {true_label:20s} | Tahmin: {pred_label:20s} | Güven: {confidence:.1f}%")
    
    print(f"\n🎉 Değerlendirme tamamlandı! Raporlar 'reports/' klasöründe.")


def predict_single(video_path, model=None):
    """
    Tek bir video dosyası için tahmin yapar.

    Args:
        video_path: Video (mp4) dosyasının yolu
        model: Keras modeli (None ise yükler)

    Returns:
        tuple: (tahmin_kelime, güven_skoru)
    """
    if model is None:
        model_path = os.path.join(project_root, 'models', 'best_model.keras')
        model = keras.models.load_model(model_path)

    # Videodan landmark çıkar
    from src.extract_landmarks import build_landmarkers, extract_from_video
    pose_lm, hand_lm = build_landmarkers()
    X = extract_from_video(video_path, pose_lm, hand_lm)
    X = np.expand_dims(X, axis=0)  # Batch boyutu ekle → (1, 30, 225)

    # Tahmin
    probs = model.predict(X, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    confidence = probs[pred_id]
    word = ID_TO_WORD.get(pred_id, f"Bilinmeyen (ID: {pred_id})")

    return word, confidence


if __name__ == "__main__":
    evaluate_model()