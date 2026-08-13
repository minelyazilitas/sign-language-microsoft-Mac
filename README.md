# SignFlow

LSA64 veri seti (64 isaret) uzerinde calisan bir isaret dili tanima projesi.
Videolardan el ve vucut noktalari cikarilir, bir BiLSTM modeli bu hareketlere
bakarak isareti tahmin eder.

## Nasil calisir

1. `extract.py` - videolari 30 kareye indirir, her kareden 225 koordinat cikarir
2. `train.py` - koordinatlarla BiLSTM modelini egitir
3. `predict.py` - modeli test setinde dener
4. `app.py` - basit bir demo arayuzu

## Kurulum

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Calistirma

```bash
python extract.py     # video -> koordinat (bir kez)
python train.py       # modeli egit
python predict.py     # test sonucu
streamlit run app.py  # demo
```

## Klasorler

- `data/videos/` - LSA64 videolari
- `data/islenmis/` - cikarilan koordinatlar (X_*.npy, y_*.npy)
- `models/` - MediaPipe model dosyalari ve egitilmis model
- `sonuclar/` - egitim grafikleri
