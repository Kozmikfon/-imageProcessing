import os
from ultralytics import YOLO

# 1. DİNAMİK YOL KATMANI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Zip'ten çıkan gerçek veri haritasını (data.yaml) doğrudan hedef gösteriyoruz
YAML_PATH = os.path.join(BASE_DIR, "datasets", "cop_seti", "data.yaml")

print("🧠 Dinamik Yapay Zeka Atık ve Çukur Eğitimi Başlatılıyor...")
print(f"📋 Okunan Veri Haritası: {YAML_PATH}\n")

# 2. MODEL MİMARİSİ
# Bilgisayarındaki yolov8n.pt temel katman ağırlık dosyasını yükler
model =YOLO('runs/detect/atik_modeli_final-2/weights/last.pt')

# 3. DİNAMİK EĞİTİM MOTORU
# Model, data.yaml dosyasındaki resimleri tek tek dinamik olarak okur
results = model.train(
    resume=True,
    data=YAML_PATH,
    epochs=30,            # 50 yerine 20 yap (zaten %80 yol aldık)
    imgsz=320,
    device='cpu',         
    batch=2,              # Bilgisayarın kasmaması için en düşük değer
    workers=0,            
    name="atik_modeli_final"
)

print("\n🎉 TEBRİKLER MAHMUT! Gerçek veri setiyle dinamik eğitim başarıyla tamamlandı.")
print(f"📁 Akıllı beyin ağırlıkları (best.pt): {os.path.join(BASE_DIR, 'runs', 'atik_modeli', 'weights', 'best.pt')} konumuna kaydedildi.")