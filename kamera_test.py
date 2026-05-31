from ultralytics import YOLO
import cv2

# Eğittiğimiz "Akıllı Beyin"
model = YOLO(r"D:\VsCode\Goruntu\runs\atik_modeli\weights\best.pt")

# Bilgisayarın kamerasını aç
cap = cv2.VideoCapture(0)

print("🎥 Penceresiz Canlı Tarama Başladı (Kapatmak için Ctrl+C yap)...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Görüntü üzerinde anlık analiz yap
        results = model(frame)
        
        # Tespit edilen nesneleri terminale yazdır
        for r in results:
            if len(r.boxes) > 0:
                print(f"🎯 Nesne Tespit Edildi: {r.names[int(r.boxes.cls[0])]}")

except KeyboardInterrupt:
    print("Durduruldu.")
    cap.release()