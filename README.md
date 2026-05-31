# 🗑️ Atık Tespit Sistemi — YOLOv8

Gerçek zamanlı kamera görüntüsü üzerinde atık (çöp) tespiti yapan derin öğrenme projesi.

## 🔧 Kullanılan Teknolojiler

- **Model:** YOLOv8n (Ultralytics)
- **Dil:** Python 3
- **Kütüphaneler:** `ultralytics`, `opencv-python`, `matplotlib`

## 📦 Veri Seti

| Küme | Görsel Sayısı |
|------|--------------|
| Train | 4730 |
| Validation | 629 |
| Sınıf Sayısı | 1 (`garbage`) |

Kaynak: [Roboflow — Garbage Detection](https://universe.roboflow.com/mahmut-baglama/garbage-detection-pbcjq-f7w4d/dataset/1)  
Lisans: CC BY 4.0

## 📈 Eğitim Parametreleri

| Parametre | Değer | Neden? |
|-----------|-------|--------|
| `epochs` | 20 | Kısıtlı donanımda yeterli yakınsama |
| `imgsz` | 320 | CPU'da hız/doğruluk dengesi |
| `batch` | 2 | Düşük RAM kullanımı |
| `device` | CPU | GPU yokluğu |

## 📊 Model Performansı (20 Epoch Sonunda)

| Metrik | Değer |
|--------|-------|
| Precision | ~%58 |
| Recall | ~%45 |
| mAP@50 | ~%47 |
| mAP@50-95 | ~%23 |

> Train ve validation loss grafikleri düzenli düşüş gösterdi — model sağlıklı öğrendi, overfit gözlemlenmedi.

## 🚀 Nasıl Çalıştırılır?

### Gereksinimler
```bash
pip install ultralytics opencv-python
```

### Canlı Kamera Testi (Sunum Modu)
```bash
python test_sunum.py
```
> Çıkmak için `Q` tuşuna bas.

### Sadece Terminal Çıktısı
```bash
python kamera_test.py
```

### Model Eğitimi
```bash
python train.py
```

## 📁 Proje Yapısı

```
Goruntu/
├── cop_modeli.pt        # Eğitilmiş model ağırlıkları
├── test_sunum.py        # Sunum modu (FPS + sınıf göstergeli)
├── kamera_test.py       # Terminal çıktılı basit test
├── train.py             # Model eğitim scripti
└── datasets/
    └── cop_seti/
        ├── train/       # 4730 eğitim görseli
        ├── valid/       # 629 doğrulama görseli
        └── data.yaml
```
