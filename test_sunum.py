from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Profesyonel isimlendirdiğimiz modelimiz
model = YOLO('cop_modeli.pt')
cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()
    
    ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    plt.pause(0.001)
    ax.clear()