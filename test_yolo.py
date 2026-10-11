from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/chicken_model-3/weights/best.pt")

# Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # 🔥 Higher confidence threshold
    results = model(frame, conf=0.7)

    chicken_detected = False

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            confidence = float(box.conf[0])

            print(f"Detected: {name} ({confidence:.2f})")

            # Only accept strong detections
            if name == "chicken" and confidence > 0.7:
                chicken_detected = True

    # Door logic
    if chicken_detected:
        print("🚪 OPEN DOOR")
    else:
        print("🔒 CLOSE DOOR")

    # Show detections
    annotated = results[0].plot()

    cv2.imshow("Smart Chicken Coop", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()