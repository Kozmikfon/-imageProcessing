from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model_path = r"D:\VsCode\Goruntu\runs\detect\atik_modeli_final-2\weights\best.pt"
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

# Matplotlib penceresini bir kez oluştur
plt.ion()
fig, ax = plt.subplots()

print("--- Canlı Tespit Başlıyor! ---")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # RGB dönüşümü
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        ax.imshow(frame_rgb)
        ax.axis('off')
        plt.draw()
        plt.pause(0.001)
        ax.clear() # Sadece eksenleri temizle
except KeyboardInterrupt:
    print("Durduruldu.")
finally:
    cap.release()
    plt.close()