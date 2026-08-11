"""
SignFlow — Veri Ön İşleme Modülü
Parquet dosyalarından landmark verilerini çıkarır ve .npy matrislerine dönüştürür.
"""

import os
import sys
import pandas as pd
import numpy as np

# --- Sabitler ---
FIXED_FRAMES = 30          # Her örnek sabit 30 frame olacak
FEATURES_PER_FRAME = 225   # 75 landmark × 3 eksen (x, y, z)
LANDMARK_TYPES = ['left_hand', 'right_hand', 'pose']  # Kullanılacak landmark tipleri


def load_single_parquet(file_path):
    """
    Tek bir parquet dosyasını okur ve (FIXED_FRAMES, FEATURES_PER_FRAME) boyutunda
    bir NumPy matrisine dönüştürür.
    
    Args:
        file_path: Parquet dosyasının yolu
        
    Returns:
        np.ndarray: (30, 225) boyutunda landmark matrisi
    """
    df = pd.read_parquet(file_path)
    
    # Sadece el ve poz landmark'larını filtrele
    filtered_df = df[df['type'].isin(LANDMARK_TYPES)]
    
    # NaN değerleri 0 ile doldur
    filtered_df = filtered_df.fillna(0)
    
    # Frame'leri sırala
    frames = sorted(filtered_df['frame'].unique())
    
    sequence = []
    for f in frames[:FIXED_FRAMES]:
        frame_df = filtered_df[filtered_df['frame'] == f]
        coords = frame_df[['x', 'y', 'z']].values.flatten()
        
        # Sabit boyuta getir (225 feature)
        if len(coords) < FEATURES_PER_FRAME:
            coords = np.pad(coords, (0, FEATURES_PER_FRAME - len(coords)), 'constant')
        elif len(coords) > FEATURES_PER_FRAME:
            coords = coords[:FEATURES_PER_FRAME]
        
        sequence.append(coords)
    
    # Frame sayısı 30'dan azsa sıfır ile doldur
    while len(sequence) < FIXED_FRAMES:
        sequence.append(np.zeros(FEATURES_PER_FRAME))
    
    return np.array(sequence, dtype=np.float32)


def prepare_dataset(csv_path, output_dir, prefix):
    """
    Bir CSV dosyasını okur, tüm parquet dosyalarını işler ve .npy dosyalarına kaydeder.
    
    Args:
        csv_path: CSV dosyasının yolu (train.csv, val.csv, test.csv)
        output_dir: Çıktı dizini (data/processed/)
        prefix: Dosya ön eki ('train', 'val', 'test')
    """
    print(f"\n{'='*60}")
    print(f"  {prefix.upper()} verisi işleniyor: {csv_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(csv_path):
        print(f"HATA: {csv_path} dosyası bulunamadı!")
        return
    
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Toplam {total} kayıt bulundu.")
    
    X = []
    y = []
    skipped = 0
    
    for index, row in df.iterrows():
        path = row['path']
        label = row['sign']
        
        if os.path.exists(path):
            try:
                seq_matrix = load_single_parquet(path)
                X.append(seq_matrix)
                y.append(label)
            except Exception as e:
                print(f"  Uyarı: {path} işlenirken hata: {e}")
                skipped += 1
        else:
            skipped += 1
        
        # İlerleme göster (her %10'da bir)
        processed = index + 1
        if processed % max(1, total // 10) == 0 or processed == total:
            pct = (processed / total) * 100
            print(f"  İlerleme: {processed}/{total} ({pct:.0f}%) — Başarılı: {len(X)}, Atlanan: {skipped}")
    
    if len(X) == 0:
        print("HATA: Hiç veri işlenemedi!")
        return
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Kaydet
    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, f'X_{prefix}.npy')
    y_path = os.path.join(output_dir, f'y_{prefix}.npy')
    
    np.save(x_path, X)
    np.save(y_path, y)
    
    print(f"\n  ✅ Tamamlandı!")
    print(f"  X_{prefix} boyutu: {X.shape}")
    print(f"  y_{prefix} boyutu: {y.shape}")
    print(f"  Benzersiz sınıf sayısı: {len(np.unique(y))}")
    print(f"  Kaydedildi: {output_dir}/")
    
    return X, y


def main():
    """Tüm verisetlerini işle."""
    # Proje kök dizinine git (CSV dosyaları orada)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    output_dir = os.path.join('data', 'processed')
    
    print("🚀 SignFlow — Veri Ön İşleme Başlıyor")
    print(f"   Çalışma dizini: {os.getcwd()}")
    print(f"   Çıktı dizini: {output_dir}")
    
    # Train, val ve test verilerini sırayla işle
    datasets = [
        ('train.csv', 'train'),
        ('val.csv', 'val'),
        ('test.csv', 'test'),
    ]
    
    for csv_file, prefix in datasets:
        if os.path.exists(csv_file):
            prepare_dataset(csv_file, output_dir, prefix)
        else:
            print(f"\n⚠️  {csv_file} bulunamadı, atlanıyor.")
    
    print(f"\n{'='*60}")
    print("  🎉 Tüm veri ön işleme tamamlandı!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()