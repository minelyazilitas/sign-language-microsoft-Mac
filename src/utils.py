"""
SignFlow — Yardımcı Fonksiyonlar
Etiket eşleştirme, veri yükleme, grafik çizme ve metrik raporlama.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI olmadan grafik üretmek için
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Etiket Sözlüğü (LSA64 — 64 Arjantin İşaret Dili kelimesi) ---
# Kaynak: LSA64 resmî sign tablosu (Ronchetti et al., 2016).
# Anahtar = 0-indeksli sınıf (dosya adındaki sign ID - 1).
ID_TO_WORD = {
    0: "Opaque",      1: "Red",         2: "Green",       3: "Yellow",
    4: "Bright",      5: "Light-blue",  6: "Colors",      7: "Pink",
    8: "Women",       9: "Enemy",       10: "Son",        11: "Man",
    12: "Away",       13: "Drawer",     14: "Born",       15: "Learn",
    16: "Call",       17: "Skimmer",    18: "Bitter",     19: "Sweet milk",
    20: "Milk",       21: "Water",      22: "Food",       23: "Argentina",
    24: "Uruguay",    25: "Country",    26: "Last name",  27: "Where",
    28: "Mock",       29: "Birthday",   30: "Breakfast",  31: "Photo",
    32: "Hungry",     33: "Map",        34: "Coin",       35: "Music",
    36: "Ship",       37: "None",       38: "Name",       39: "Patience",
    40: "Perfume",    41: "Deaf",       42: "Trap",       43: "Rice",
    44: "Barbecue",   45: "Candy",      46: "Chewing-gum", 47: "Spaghetti",
    48: "Yogurt",     49: "Accept",     50: "Thanks",     51: "Shut down",
    52: "Appear",     53: "To land",    54: "Catch",      55: "Help",
    56: "Dance",      57: "Bathe",      58: "Buy",        59: "Copy",
    60: "Run",        61: "Realize",    62: "Give",       63: "Find",
}

WORD_TO_ID = {v: k for k, v in ID_TO_WORD.items()}
NUM_CLASSES = len(ID_TO_WORD)


def get_project_root():
    """Proje kök dizinini döndürür."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_processed_data(prefix, data_dir=None):
    """
    İşlenmiş .npy verilerini yükler.
    
    Args:
        prefix: 'train', 'val' veya 'test'
        data_dir: Veri dizini (varsayılan: data/processed/)
        
    Returns:
        tuple: (X, y) NumPy dizileri
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root(), 'data', 'processed')
    
    x_path = os.path.join(data_dir, f'X_{prefix}.npy')
    y_path = os.path.join(data_dir, f'y_{prefix}.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"{prefix} verisi bulunamadı: {data_dir}\n"
            "Önce 'python src/preprocess.py' komutunu çalıştırın."
        )
    
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"✅ {prefix} verisi yüklendi — X: {X.shape}, y: {y.shape}")
    return X, y


def plot_training_history(history, save_dir=None):
    """
    Eğitim geçmişi grafiklerini çizer (loss ve accuracy).
    
    Args:
        history: Keras model.fit() tarafından döndürülen History nesnesi
        save_dir: Grafiklerin kaydedileceği dizin
    """
    if save_dir is None:
        save_dir = os.path.join(get_project_root(), 'reports')
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss grafiği
    axes[0].plot(history.history['loss'], label='Eğitim Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Doğrulama Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(history.history['accuracy'], label='Eğitim Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Doğrulama Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Eğitim grafikleri kaydedildi: {path}")


def plot_confusion_matrix(y_true, y_pred, save_dir=None):
    """
    Confusion matrix görselini oluşturur ve kaydeder.
    
    Args:
        y_true: Gerçek etiketler (integer)
        y_pred: Tahmin edilen etiketler (integer)
        save_dir: Kaydedilecek dizin
    """
    if save_dir is None:
        save_dir = os.path.join(get_project_root(), 'reports')
    os.makedirs(save_dir, exist_ok=True)
    
    labels = list(range(NUM_CLASSES))
    label_names = [ID_TO_WORD[i] for i in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        square=True,
        linewidths=0.5,
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Tahmin', fontsize=12)
    plt.ylabel('Gerçek', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix kaydedildi: {path}")


def print_classification_report(y_true, y_pred, save_dir=None):
    """
    Detaylı sınıflandırma raporu yazdırır ve dosyaya kaydeder.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        save_dir: Raporun kaydedileceği dizin
    """
    if save_dir is None:
        save_dir = os.path.join(get_project_root(), 'reports')
    os.makedirs(save_dir, exist_ok=True)
    
    label_names = [ID_TO_WORD[i] for i in range(NUM_CLASSES)]
    
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
    
    print("\n" + "="*60)
    print("  SINIFLANDIRMA RAPORU")
    print("="*60)
    print(report)
    
    # Dosyaya kaydet
    path = os.path.join(save_dir, 'classification_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("SignFlow — Sınıflandırma Raporu\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"📄 Rapor kaydedildi: {path}")
