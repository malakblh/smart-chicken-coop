from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO("runs/detect/chicken_model-3/weights/best.pt")

# Tester une image
results = model("test_images/poule.jpg")

# Afficher résultat
results[0].show()

# Vérifier détection
if len(results[0].boxes) > 0:
    print("🚪 OPEN DOOR - Chicken detected")
else:
    print("🔒 CLOSE DOOR - No chicken detected")