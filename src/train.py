"""
SignFlow — Model Eğitimi
BiLSTM tabanlı Türk İşaret Dili sınıflandırma modeli eğitimi.
"""

import os
import sys
import numpy as np

# Proje kök dizinini ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_processed_data, plot_training_history, NUM_CLASSES

# TensorFlow uyarılarını azalt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_bilstm_model(input_shape, num_classes):
    """
    BiLSTM tabanlı sınıflandırma modeli oluşturur.
    
    Args:
        input_shape: Girdi boyutu (timesteps, features) → (30, 225)
        num_classes: Sınıf sayısı (21)
        
    Returns:
        Derlenmiş Keras modeli
    """
    model = keras.Sequential([
        # Girdi katmanı
        layers.Input(shape=input_shape),
        
        # Sıfır padding'i yok saymak için maskeleme
        layers.Masking(mask_value=0.0),
        
        # İlk BiLSTM katmanı — sekans çıkışı döndürür
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.3)
        ),
        
        # İkinci BiLSTM katmanı — son çıkışı döndürür
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=False, dropout=0.3)
        ),
        
        # Tam bağlantılı katmanlar
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Çıkış katmanı — 21 sınıf
        layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model


def train_model():
    """Ana eğitim fonksiyonu."""
    print("🚀 SignFlow — Model Eğitimi Başlıyor")
    print("="*60)
    
    # --- 1. Verileri Yükle ---
    print("\n📂 Veriler yükleniyor...")
    X_train, y_train = load_processed_data('train')
    X_val, y_val = load_processed_data('val')
    
    print(f"\n   Eğitim seti:    {X_train.shape[0]} örnek")
    print(f"   Doğrulama seti: {X_val.shape[0]} örnek")
    print(f"   Girdi boyutu:   {X_train.shape[1:]}")
    print(f"   Sınıf sayısı:   {NUM_CLASSES}")
    
    # --- 2. Modeli Oluştur ---
    print("\n🏗️  Model oluşturuluyor...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 225)
    model = build_bilstm_model(input_shape, NUM_CLASSES)
    model.summary()
    
    # --- 3. Callback'ler ---
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    best_model_path = os.path.join(models_dir, 'best_model.keras')
    
    callbacks = [
        # En iyi modeli kaydet
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        # Erken durdurma — 10 epoch boyunca iyileşme yoksa dur
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Öğrenme hızını azalt
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    # --- 4. Eğitimi Başlat ---
    print("\n🎯 Eğitim başlıyor...")
    print("-"*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    
    # --- 5. Sonuçları Kaydet ---
    print("\n" + "="*60)
    print("  📊 EĞİTİM SONUÇLARI")
    print("="*60)
    
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"  En iyi doğrulama accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  Model kaydedildi: {best_model_path}")
    
    # Eğitim grafiklerini kaydet
    plot_training_history(history)
    
    print("\n🎉 Eğitim tamamlandı!")
    
    return model, history


if __name__ == "__main__":
    train_model()
