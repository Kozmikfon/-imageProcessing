from ultralytics import YOLO
import cv2
import time

# =============================================
#  ATIK TESPİT SİSTEMİ - SUNUM MODU
# =============================================

model = YOLO(r'runs/detect/atik_modeli_final-3/weights/best.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONF_ESIGI = 0.25

# FPS hesabı için
fps_sayac = 0
fps_goster = 0
fps_zaman = time.time()

print("=" * 50)
print("   ATIK TESPİT SİSTEMİ - SUNUM MODU")
print("=" * 50)
print("  Çıkmak için: Q tuşuna bas")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadı!")
        break

    # ---- Tespit ----
    results = model(frame, conf=CONF_ESIGI, verbose=False)
    annotated = results[0].plot()

    # ---- FPS Hesapla ----
    fps_sayac += 1
    gecen = time.time() - fps_zaman
    if gecen >= 1.0:
        fps_goster = fps_sayac / gecen
        fps_sayac = 0
        fps_zaman = time.time()

    # ---- Tespit Edilen Sınıflar ----
    boxes = results[0].boxes
    sinif_sayilari = {}
    if boxes is not None and len(boxes) > 0:
        for cls_id, conf in zip(boxes.cls, boxes.conf):
            sinif_adi = results[0].names[int(cls_id)]
            sinif_sayilari[sinif_adi] = sinif_sayilari.get(sinif_adi, 0) + 1

    # ---- HUD Çiz ----
    h, w = annotated.shape[:2]

    # Üst sol: başlık + FPS
    cv2.rectangle(annotated, (0, 0), (300, 60), (0, 0, 0), -1)
    cv2.putText(annotated, "ATIK TESPİT SİSTEMİ",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 150), 1)
    cv2.putText(annotated, f"FPS: {fps_goster:.1f}  |  Conf: {CONF_ESIGI}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Üst sağ: toplam tespit sayısı
    toplam = len(boxes) if boxes is not None else 0
    etiket = f"Toplam Tespit: {toplam}"
    (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.rectangle(annotated, (w - tw - 20, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(annotated, etiket,
                (w - tw - 10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 1)

    # Alt sol: sınıf listesi
    if sinif_sayilari:
        panel_h = len(sinif_sayilari) * 28 + 15
        cv2.rectangle(annotated, (0, h - panel_h - 5), (220, h), (0, 0, 0), -1)
        for i, (ad, sayi) in enumerate(sinif_sayilari.items()):
            y = h - panel_h + 5 + (i + 1) * 28
            cv2.putText(annotated, f"  {ad}: {sayi}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

    # Alt sağ: çıkış ipucu
    cv2.putText(annotated, "Q: Cik",
                (w - 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Atik Tespit Sistemi", annotated)

    # Q veya ESC ile çık
    tus = cv2.waitKey(1) & 0xFF
    if tus in (ord('q'), ord('Q'), 27):
        print("\nProgram kapatılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
print("Kamera kapatıldı. İyi sunumlar!")